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

from joblib import Parallel, delayed
# threads = Parallel(n_jobs=n_jobs, backend='multiprocessing')

def process_sample(i, row):
    if 'OGLE' in row['path']:
        df = pd.read_csv(row['path'], sep='\s+')
    elif 'lmc' in row['path'] or \
         'bul' in row['path'] or \
         'car' in row['path'] or \
         'smc' in row['path']: 
        df = pd.read_csv(row['path'], sep='\s+')
        df = df.iloc[:, :3]
    else:
        df = pd.read_csv(row['path'], engine='c', na_filter=False)
        
    df.columns = ['mjd', 'mag', 'errmag']
    
    df['newID'] = row.newID*np.ones(df.shape[0]).astype(np.int64)
    return df

def process_sample_2(i, row):
    df = pd.read_csv(row['path'], names=['mjd', 'mag', 'err'], sep='\s+')
    df = df.iloc[:, :3]
    df.columns = ['mjd', 'mag', 'errmag']
    df['newID'] = row.newID*np.ones(df.shape[0]).astype(np.int64)
    return df
    
def run(opt):
    
    opt.data = os.path.normpath(opt.data)
    
    metadata = pd.read_csv(os.path.join(opt.data, 'metadata.csv'))
    metadata['sset'] = ['train']*metadata.shape[0]

    if os.path.exists(os.path.join(opt.data, 'test_metadata.csv')):
        test_metadata = pd.read_csv(os.path.join(opt.data, 'test_metadata.csv'))
        test_metadata['sset'] = ['test']*test_metadata.shape[0]
        metadata = pd.concat([metadata, test_metadata])
    if opt.debug:
        metadata = metadata.sample(20)
    
    dataset_name = os.path.basename(opt.data)
    target_dir = os.path.join(opt.target, dataset_name)
    light_curves_dir = os.path.join(target_dir, 'light_curves')
    os.makedirs(light_curves_dir, exist_ok=True)

    metadata = metadata.assign(newID=range(len(metadata)))
    metadata['path'] = (opt.data + '/LCs/' + metadata.ID.astype(str)+'.dat').to_list()

    threads = Parallel(n_jobs=opt.n_jobs, backend='loky')

    dfs = threads(delayed(process_sample)(i, row) for i, row in metadata.iterrows()) 

    ids_parquet = []
    for batch, begin in enumerate(np.arange(0, len(dfs), opt.samples_per_chunk)):
        df_sel = dfs[begin:begin+opt.samples_per_chunk]    
        df_sel = pd.concat(df_sel)
        ids = df_sel.newID.unique()
        ids_parquet.append(pd.DataFrame({'newID': ids, 'shard': [batch]*len(ids)}))
        n = str(batch).rjust(3, '0')
        df_sel.to_parquet(os.path.join(light_curves_dir, 'shard_'+n+'.parquet'))
    

    partial = pd.concat(ids_parquet)
    metadata['Label'] = pd.Categorical(metadata['Class']).codes
    metadata = pd.merge(metadata, partial, on='newID')
    metadata.to_parquet(os.path.join(target_dir, 'metadata.parquet'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/raw_data/alcock', type=str,
                    help='raw_data folder')
    parser.add_argument('--target', default='./data/praw/', type=str,
                    help='target folder to save parquet files')

    parser.add_argument('--samples-per-chunk', default=20000, type=int,
                    help='Number of light curves per chunk')
    parser.add_argument('--n-jobs', default=1, type=int,
                    help='Number of cores')

    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


    opt = parser.parse_args()        
    run(opt)
