import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import toml
import sys
import os
from tqdm import tqdm

from src.utils import get_metrics
from presentation.pipelines.steps.model_design import load_pt_model
from presentation.pipelines.steps.load_data import build_loader 
from src.training.utils import test_step


def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    astromer, config = load_pt_model(opt.model)
    print('[INFO] {} loaded'.format(opt.model))
    
    df = pd.DataFrame(config, index=[0])
    loaders = build_loader(df['data'].values[0], 
                           config, 
                           batch_size=opt.bs,
                           clf_mode=False,
                           sampling=False,
                           return_test=True,
                           normalize='zero-mean')  
    
    test_rmse = 0.
    test_rsquare = 0.
    batch_counter = 0
    for batch in tqdm(loaders['test']):
        metrics = test_step(astromer, batch)
        test_rmse+=metrics['rmse'].numpy()
        test_rsquare+=metrics['rsquare'].numpy()
        batch_counter+=1

    test_rmse = test_rmse/batch_counter
    test_rsquare = test_rsquare/batch_counter

    
    valid_loss = get_metrics(os.path.join(opt.model, 'tensorboard', 'validation'), 
                                metric_name='rmse')
    
    best_loss = valid_loss[valid_loss['value']==valid_loss['value'].min()]

    valid_rsquare = get_metrics(os.path.join(opt.model, 'tensorboard', 'validation'), 
                                metric_name='rsquare')
    
    metrics = {
        'test_r2': [test_rsquare],
        'test_mse': [test_rmse],
        'val_mse': [float(best_loss['value'].values[0])],
        'val_r2': [float(valid_rsquare.iloc[best_loss.index]['value'].values[0])]
    }
    metrics = pd.DataFrame(metrics)
    df = pd.concat([df, metrics], axis=1)
    df.to_csv(os.path.join(opt.model, 'results.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/presentation/results/model', type=str,
                    help='Model folder')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU Device')
    parser.add_argument('--bs', default=2500, type=int,
                        help='Batch size')



    opt = parser.parse_args()        
    run(opt)
