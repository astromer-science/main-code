'''
This script was made exclusively to transform old data format to a new parquet-based one.
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import toml
import sys
import os

from src.data.record import DataPipeline, create_config_toml

import time
import polars as pl

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class CustomCleanPipeline(DataPipeline):
    def lightcurve_step(self, inputs):
        """
        Preprocessing applied to each light curve separately
        """
        # First feature is time
        inputs = inputs.sort(self.sequential_features[0], descending=True) 
        p99 = inputs.quantile(0.99, 'nearest')
        p01 = inputs.quantile(0.01, 'nearest')
        inputs = inputs.filter(pl.col('mag') < p99['mag'])
        inputs = inputs.filter(pl.col('mag') > p01['mag'])
        return inputs

    def observations_step(self):
        """
        Preprocessing applied to all observations. Filter only
        """
        fn_0 = pl.col("err") < 1.  # Clean the data on the big lazy dataframe
        fn_1 = pl.col("err") > 0.
        fn_2 = pl.col("Band") == 'R'

        return fn_0 & fn_1 & fn_2

def run(opt):
    
    start = time.time()
    if os.path.exists(opt.config):
        print('[INFO] Loading config.toml at {}'.format(opt.config))
        with open(opt.config) as handle:
            config = toml.load(handle)

    METAPATH   = config['context_features']['path']
    TARGETPATH = config['target']['path']

    metadata = pd.read_parquet(METAPATH)
    
    metadata['subset_0'] = ['train']*metadata.shape[0]
        
    pipeline = CustomCleanPipeline(metadata=metadata,
                                   config_path=opt.config)
    
    var = pipeline.run(n_jobs=opt.njobs,
                       elements_per_shard=opt.elements_per_shard)

    end = time.time()
    print('\n [INFO] ELAPSED: ', end - start)

if __name__ == '__main__':
    # python -m presentation.scripts.create_records --config ./data/my_data_folder/config.toml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./data/raw_data_parquet/alcock/config.toml', type=str,
                    help='Config file specifying context and sequential features')

    parser.add_argument('--njobs', default=4, type=int,
                    help='Number of cores to use')
    
    parser.add_argument('--elements-per-shard', default=20000, type=int,
                    help='Number of light curves per shard')


    opt = parser.parse_args()        
    run(opt)
