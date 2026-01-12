import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import toml
import sys
import os
from tqdm import tqdm

# Import utilities
from src.utils import get_metrics
from presentation.pipelines.steps.model_design import load_pt_model
from presentation.pipelines.steps.load_data import build_loader 
from src.training.utils import test_step

def get_training_duration(model_folder):
    """
    Calculates the training duration (wall time) using TensorBoard logs.
    It reads the 'rmse' metric from the validation folder to determine start and end times.
    """
    try:
        # Path to validation logs
        log_dir = os.path.join(model_folder, 'tensorboard', 'validation')
        
        # Use existing utility to get the DataFrame [wall_time, step, value]
        # We assume 'rmse' exists as it is the standard metric in this pipeline
        df = get_metrics(log_dir, metric_name='rmse')
        
        if df is not None and not df.empty:
            start_time = df['wall_time'].min()
            end_time = df['wall_time'].max()
            duration_seconds = end_time - start_time
            return duration_seconds
        else:
            return 0.0
    except Exception as e:
        print(f"[WARNING] Could not calculate training duration: {e}")
        return 0.0

def evaluate_loop(model, dataset, desc="Evaluating"):
    """
    Helper function to iterate over the dataset and calculate metrics.
    """
    test_rmse = 0.
    test_rsquare = 0.
    batch_counter = 0

    print(f'[INFO] {desc}...')
    # Removed .take(10) to evaluate the full dataset as requested for production
    for batch in tqdm(dataset):
        metrics = test_step(model, batch)
        test_rmse += metrics['rmse'].numpy()
        test_rsquare += metrics['rsquare'].numpy()
        batch_counter += 1

    if batch_counter > 0:
        return (test_rmse / batch_counter), (test_rsquare / batch_counter)
    return 0., 0.

def run(opt):
    """
    Main execution function.
    1. Loads the finetuned model and config.
    2. Builds the data loader (custom or from config).
    3. Calculates training time from logs.
    4. Evaluates the Finetuned Model.
    5. (Optional) Evaluates the Base Model (pre-finetuning).
    6. Saves results.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    # --- 1. Load Finetuned Model & Config ---
    astromer, config = load_pt_model(opt.model)
    print(f'[INFO] Finetuned model loaded from: {opt.model}')
    
    # Create base DataFrame with the model configuration
    df = pd.DataFrame(config, index=[0])

    # --- 2. Determine Dataset ---
    if opt.dataset is not None:
        print(f'[INFO] Using custom dataset: {opt.dataset}')
        data_path = opt.dataset
        df['data'] = data_path
        output_name = 'results_custom.csv'
        base_output_name = 'results_base_custom.csv'
    else:
        print(f'[INFO] Using dataset from configuration: {config["data"]}')
        data_path = config['data']
        output_name = 'results.csv'
        base_output_name = 'results_base.csv'

    # Configuration overrides for reconstruction
    config['probed'] = 1.0
    config['rs'] = 0.
    config['same'] = 0.

    # --- 3. Build Loader ---
    loaders = build_loader(data_path, 
                           config, 
                           batch_size=opt.bs,
                           clf_mode=False,
                           sampling=False,
                           return_test=True,
                           normalize='zero-mean')  
    
    if 'test' not in loaders:
        print("[ERROR] Test data not found in loader.")
        return

    # --- 4. Get Training Duration ---
    # Extract walltime from TensorBoard logs
    training_duration = get_training_duration(opt.model)
    print(f"[INFO] Extracted Training Duration: {training_duration:.2f} seconds")

    # --- 5. Evaluate Finetuned Model ---
    ft_rmse, ft_r2 = evaluate_loop(astromer, loaders['test'], desc="Evaluating Finetuned Model")

    # Get validation metrics from TensorBoard (Best epoch)
    try:
        valid_loss = get_metrics(os.path.join(opt.model, 'tensorboard', 'validation'), metric_name='rmse')
        best_loss = valid_loss[valid_loss['value'] == valid_loss['value'].min()]
        
        valid_rsquare = get_metrics(os.path.join(opt.model, 'tensorboard', 'validation'), metric_name='rsquare')
        
        val_mse_val = float(best_loss['value'].values[0])
        val_r2_val = float(valid_rsquare.iloc[best_loss.index]['value'].values[0])
    except Exception as e:
        print(f"[WARNING] Could not load previous validation metrics: {e}")
        val_mse_val = -1
        val_r2_val = -1
    
    # Construct metrics dictionary
    metrics = {
        'test_r2': [ft_r2],
        'test_mse': [ft_rmse],
        'val_mse': [val_mse_val],
        'val_r2': [val_r2_val],
        'training_time': [training_duration]  # New Column
    }
    
    metrics_df = pd.DataFrame(metrics)
    final_df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)
    
    save_path = os.path.join(opt.model, output_name)
    final_df.to_csv(save_path, index=False)
    print(f'[INFO] Finetuned results saved to: {save_path}')

    # --- 6. (Optional) Evaluate Base Model ---
    if opt.base_model is not None:
        print(f"------------------------------------------------")
        print(f"[INFO] Loading Base Model (No Finetuning) from: {opt.base_model}")
        
        try:
            # Load base model
            # We discard the config here because we want to compare using the dataset 
            # and hyperparameters defined in the current run, but with old weights.
            base_astromer, _ = load_pt_model(opt.base_model)
            
            # Evaluate using the SAME loader
            base_rmse, base_r2 = evaluate_loop(base_astromer, loaders['test'], desc="Evaluating Base Model")
            
            # Create a simplified results file for the base model
            base_metrics = {
                'model_path': [opt.base_model],
                'dataset_used': [data_path],
                'test_r2': [base_r2],
                'test_mse': [base_rmse]
            }
            base_df = pd.DataFrame(base_metrics)
            
            # Save in the SAME folder as the finetuned model for comparison
            base_save_path = os.path.join(opt.model, base_output_name)
            base_df.to_csv(base_save_path, index=False)
            print(f'[INFO] Base model results saved to: {base_save_path}')
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate base model: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ASTROMER model on a dataset')
    
    parser.add_argument('--model', default='/presentation/results/model', type=str,
                    help='Path to the finetuned model folder')
    
    parser.add_argument('--base_model', default=None, type=str,
                    help='(Optional) Path to the base model (pre-trained) to evaluate performance before finetuning.')

    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU Device ID (e.g., "0" or "-1" for CPU)')
    
    parser.add_argument('--bs', default=2500, type=int,
                        help='Batch size')

    parser.add_argument('--dataset', default=None, type=str,
                        help='Optional path to an external dataset (tfrecords or folder).')

    opt = parser.parse_args()        
    run(opt)