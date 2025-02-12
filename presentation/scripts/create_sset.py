'''
This script was made exclusively to transform old data format to a new parquet-based one.
'''

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

import time
import polars as pl

def run(opt):
    start = time.time()
    if os.path.exists(opt.config):
        print('[INFO] Loading config.toml at {}'.format(opt.config))
        with open(opt.config) as handle:
            config = toml.load(handle)

    METAPATH   = config['context_features']['path']
    TARGETPATH = config['target']['path']
    
    metadata = pd.read_parquet(METAPATH)
        
    got_test = False
    if os.path.exists(config['context_features']['test_path']):
        print('[INFO] Loading test metadata at {}'.format(config['context_features']['test_path']))  
        test_metadata = pd.read_parquet(config['context_features']['test_path'])
        metadata = pd.concat([test_metadata, metadata])
        got_test = True
    
    for n_samples in [1000, 10000, 100000, 200000, 400000]:
                
        rest = metadata[~metadata['newID'].isin(test_metadata['newID'])]
        assert test_metadata['newID'].isin(rest['newID']).sum() == 0 # check if there are duplicated indices
    
        # ==== VALIDATION/TRAIN DATA ==============================
        currmeta = rest.sample(n=n_samples)
        
        validation_metadata = currmeta.sample(frac=opt.val_frac)
        train_metadata      = currmeta[~currmeta['newID'].isin(validation_metadata['newID'])]
        assert train_metadata['newID'].isin(validation_metadata['newID']).sum() == 0 # check if there are duplicated indices

        # ==== FOLD =======================================
        train_metadata['subset_0']      = ['train']*train_metadata.shape[0]
        validation_metadata['subset_0'] = ['validation']*validation_metadata.shape[0]
        test_metadata['subset_0']       = ['test']*test_metadata.shape[0]

        final_meta = pd.concat([train_metadata, validation_metadata, test_metadata], axis=0)        
        
        a = final_meta[final_meta['subset_0'] == 'train']
        b = final_meta[final_meta['subset_0'] == 'validation']
        c = final_meta[final_meta['subset_0'] == 'test']
        print('train: {} - val: {} - test: {}'.format(a.shape, b.shape, c.shape) )       
        
        config['target']['path'] = os.path.join('./data/records/macho', str(n_samples))
        os.makedirs(config['target']['path'], exist_ok=True)
        final_meta.to_parquet(os.path.join(config['target']['path'], 'metadata.parquet'), index=False)   
        
        config['context_features']['path'] = os.path.join(config['target']['path'], 'metadata.parquet')
        config_curr = os.path.join(config['target']['path'], 'config.toml')
        with open(config_curr, 'w') as file:
            toml.dump(config, file)
        
        pipeline = DataPipeline(metadata=final_meta,
                                config_path=config_curr)
    
        var = pipeline.run(n_jobs=opt.njobs,
                           elements_per_shard=opt.elements_per_shard)
   
        end = time.time()
        print('\n [INFO] ELAPSED: ', end - start)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./data/raw_data/macho/config.toml', type=str,
                    help='Config file specifying context and sequential features')

    parser.add_argument('--val-frac', default=0.2, type=float,
                    help='Validation fraction')
    parser.add_argument('--test-frac', default=0.2, type=float,
                    help='Validation fraction')

    parser.add_argument('--njobs', default=4, type=int,
                    help='Number of cores to use')


    parser.add_argument('--elements-per-shard', default=20000, type=int,
                    help='Number of light curves per shard')


    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


    opt = parser.parse_args()        
    run(opt)
