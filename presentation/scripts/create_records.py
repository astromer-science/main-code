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
    
    # if there is a column associated with the testing subset, and split the dataframe 
    # got_test = true
    
    for fold_k in range(opt.folds):
    
        # ==== TEST DATA ==========================================
        if not got_test:
            test_metadata = metadata.sample(frac=opt.test_frac)
            
        rest = metadata[~metadata['newID'].isin(test_metadata['newID'])]
        assert test_metadata['newID'].isin(rest['newID']).sum() == 0 # check if there are duplicated indices
    
        # ==== VALIDATION/TRAIN DATA ==============================
        validation_metadata = rest.sample(frac=opt.val_frac)
        train_metadata = rest[~rest['newID'].isin(validation_metadata['newID'])]
        assert train_metadata['newID'].isin(validation_metadata['newID']).sum() == 0 # check if there are duplicated indices
    
        # ==== FOLD =======================================
        train_metadata['subset_{}'.format(fold_k)] = ['train']*train_metadata.shape[0]
        validation_metadata['subset_{}'.format(fold_k)] = ['validation']*validation_metadata.shape[0]
        test_metadata['subset_{}'.format(fold_k)] = ['test']*test_metadata.shape[0]

        curr_meta = pd.concat([train_metadata, validation_metadata, test_metadata], axis=0)
        cols_to_use = curr_meta.columns.difference(metadata.columns)
        metadata = pd.merge(metadata, curr_meta[cols_to_use], left_index=True, right_index=True, how='outer')

        
    pipeline = DataPipeline(metadata=metadata,
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

    parser.add_argument('--folds', default=1, type=int,
                    help='number of folds')
    parser.add_argument('--val-frac', default=0.2, type=float,
                    help='Validation fraction')
    parser.add_argument('--test-frac', default=0.2, type=float,
                    help='Validation fraction')

    parser.add_argument('--njobs', default=4, type=int,
                    help='Number of cores to use')
    
    parser.add_argument('--elements-per-shard', default=200000, type=int,
                    help='Number of light curves per shard')


    opt = parser.parse_args()        
    run(opt)
