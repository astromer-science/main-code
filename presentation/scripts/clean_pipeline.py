import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import polars as pl
import glob
import toml
import sys
import os

from src.data.record import DataPipeline, create_config_toml
from joblib import Parallel, delayed


import time
import polars as pl


def clean_function(inputs):
    idx = inputs[0]
    frame = inputs[1]
            
    cond_0 = frame['mag'].skew() > 1
    cond_1 = frame['mag'].kurtosis() > 10
    cond_2 = frame['mag'].std() > 0.1
    
    if cond_0 and cond_1 and cond_2:
        return None
    else:
        return idx

def run(opt):
    
    metadata = pd.read_parquet(os.path.join(opt.data, 'metadata.parquet'))
    n_init = metadata.shape[0]
    parquet_shards = glob.glob(os.path.join(opt.data, 'light_curves', '*.parquet'))
    
    cond = []
    for pfile in parquet_shards:
        observations = pd.read_parquet(pfile)
        var = Parallel(n_jobs=opt.njobs)
        outputs = var(delayed(clean_function)(x) for x in observations.groupby('newID'))        
        outputs = [x for x in outputs if x is not None]
        n = metadata.shape[0]
        metadata = metadata[~metadata['newID'].isin(outputs)]

    n_final = metadata.shape[0]
    print('[INFO] {}/{} dropped from original metadata'.format(n_init - n_final, n_init))
    metadata.to_parquet(os.path.join(opt.data, 'cleaned_metadata.parquet'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/raw_data/macho/', type=str,
                    help='light curves folder')
    
    parser.add_argument('--njobs', default=4, type=int,
                    help='Number of cores to use')
    parser.add_argument('--elements-per-shard', default=20000, type=int,
                    help='Number of light curves per shard')
    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


    opt = parser.parse_args()        
    run(opt)