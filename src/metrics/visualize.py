import matplotlib.pyplot as plt
import pandas as pd
import toml
import os
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import get_metrics

from glob import glob
from sklearn.metrics import confusion_matrix
from tensorboard.backend.event_processing import event_accumulator


def get_saved_metrics(path):
	with open(path, 'r') as f:
		m = toml.load(f)
		return m

def get_keys(path_logs):
    train_logs = [x for x in os.listdir(path_logs) if x.endswith('.v2')][0]
    path_train = os.path.join(path_logs, train_logs)

    ea = event_accumulator.EventAccumulator(path_train, 
                                            size_guidance={'tensors': 0})

      
    ea.Reload()

    keys = [x for x in ea.Tags()['tensors'] if 'epoch' in x]
    return keys

def get_validation_metrics(path, tag=None):
	valid_path = os.path.join(path, 'validation')
	keys = get_keys(valid_path)
	for i, k in enumerate(keys):
		df = get_metrics(valid_path, metric_name=k, full_logs=True)
		
		df = df.rename(columns={'value': k})
		if i == 0 :
			frame = df
		else:
			df = df.drop('wall_time', axis=1)
			frame = pd.merge(frame, df, how='right', on=['step'])

	if tag is not None:
		for i, t in enumerate(tag.split('_')):
			frame['col_{}'.format(i)] = [t]*frame.shape[0]
	return frame


def get_training_metrics(path, tag=None):
	train_path = os.path.join(path, 'train')

	keys = get_keys(train_path)
	for i, k in enumerate(keys):
		df = get_metrics(train_path, metric_name=k, full_logs=True)

		df = df.rename(columns={'value': k})
		if i == 0 :
			frame = df
		else:
			df = df.drop('wall_time', axis=1)
			frame = pd.merge(frame, df, how='right', on=['step'])

	if tag is not None:
		for i, t in enumerate(tag.split('_')):
			frame['col_{}'.format(i)] = [t]*frame.shape[0]
	return frame

def compute_cm_stats(root, dataname, spc_list=[20, 100, 500]):
    """
    Loads predictions and calculates Mean and Std Confusion Matrices for each SPC.
    
    Returns:
        dict: {spc: {'mean': mean_cm, 'std': std_cm}}
    """
    stats = {}
    
    for spc in spc_list:
        # Search pattern based on original code
        pattern = os.path.join(root, 'classification', dataname.lower(), 
                               '*', '*_{}'.format(spc), 'skip_avg_mlp', '*pkl')
        paths = glob(pattern)
        
        if not paths:
            print(f"Warning: No files found for {dataname} SPC {spc} at {pattern}")
            continue
            
        confmatrices = []
        for p in paths:
            with open(p, 'rb') as handle:
                predictions = pickle.load(handle)
                y_true = np.argmax(predictions['true'], 1)
                y_pred = np.argmax(predictions['pred'], 1)
                cm = confusion_matrix(y_true, y_pred, normalize='true')
                confmatrices.append(cm)
        
        # Calculate stats
        confmatrices = np.array(confmatrices)
        stats[spc] = {
            'mean': np.mean(confmatrices, 0),
            'std': np.std(confmatrices, 0)
        }
    
    return stats

def _get_metric_value(path, column_name, filename='training.log'):
    """
    Helper interno para leer el primer valor de una métrica desde un CSV.
    """
    log_path = os.path.join(path, filename)
    try:
        df = pd.read_csv(log_path)
        if column_name in df.columns:
            return df[column_name].iloc[0]
    except Exception:
        pass
    return None

def get_sorted_experiments(model_paths, tag='base'):
    """
    Ordena las rutas de los modelos y genera etiquetas basadas en el TAG especificado.

    Args:
        model_paths (list): Lista de rutas a las carpetas de los experimentos.
        tag (str): Criterio de ordenamiento ('diagstromer', 'temperature', 'm_alpha').

    Returns:
        tuple: (sorted_paths, sorted_labels)
    """

    experiments_data = []

    for path in model_paths:
        label = "Unknown"
        sort_key = "" 
        try:
            with open(os.path.join(path, 'config.toml'), 'r') as file:
                conf = toml.load(file)
        except FileNotFoundError:
            print(f"Warning: config.toml not found in {path}")
            continue

        if tag == 'base':
            if 'bigmacho' in conf['data']:
                label = 'A2x'
            else:
                label = 'A2'
            sort_key = label

        elif tag == 'temperature':
            val = _get_metric_value(path, 'temperature')
            if val is not None:
                label = r'$\tau$={:.2f}'.format(val)
                sort_key = val
            else:
                sort_key = -1 # Default fallback

        elif tag == 'm_alpha':
            val = _get_metric_value(path, 'm_alpha')
            if val is not None:
                if float(val) == -1e9:
                    label = r'$\alpha$=$-\infty$'
                    sort_key = -float('inf')
                else:
                    label = r'$\alpha$={:.0f}'.format(val)
                    sort_key = val
            else:
                sort_key = 0

        experiments_data.append({
            'sort_key': sort_key,
            'label': label,
            'path': path
        })

    experiments_data.sort(key=lambda x: x['sort_key'])
    sorted_paths = [x['path'] for x in experiments_data]
    sorted_labels = [x['label'] for x in experiments_data]

    return sorted_paths, sorted_labels

def load_experiment_metrics(paths, loader_func):
    """
    Iterates through paths and loads training/validation metrics using the provided loader function.

    Args:
        paths (list): List of directory paths.
        loader_func (callable): Function that takes a path and returns (train_df, val_df).
                                e.g., your 'tensorboard_logs' function.

    Returns:
        tuple: (train_metrics_list, val_metrics_list)
    """
    train_metrics = []
    val_metrics = []
    
    for path in paths:
        try:
            tmr, vmr = loader_func(path)
            train_metrics.append(tmr)
            val_metrics.append(vmr)
        except Exception as e:
            print(f"Error loading logs for {path}: {e}")
            # Append None or empty DF to maintain index alignment
            train_metrics.append(None)
            val_metrics.append(None)
            
    return train_metrics, val_metrics