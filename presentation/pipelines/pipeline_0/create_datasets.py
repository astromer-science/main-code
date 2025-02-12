import pandas as pd 
import argparse
import toml 
import time
import os

from src.data.record import DataPipeline

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def run(opt):
    data = opt.data
    
    metapath = os.path.join(opt.data, 'metadata.parquet')
        
    lcspath  = os.path.join(opt.data, 'light_curves')
    basename = os.path.basename(os.path.normpath(opt.target))

    with open(opt.ref_config, 'r') as file:
        config = toml.load(file)
    
    for spc in [20, 100, 500, '']:    
        start = time.time()
        metadata = pd.read_parquet(metapath)
        metadata = metadata[metadata['Class'] != 'Dubious']
                
        for nfold in range(opt.folds):
            fold_path = os.path.join(opt.target, str(spc), 'fold_{}'.format(nfold))
            os.makedirs(fold_path, exist_ok=True)
            
            test_metadata = metadata.groupby('Label').sample(n=opt.ntest)
            rest_metadata = metadata[~metadata['newID'].isin(test_metadata['newID'])]
                        
            # Train Val and Test splits
            if spc != '':
                nval = int(opt.val_frac*spc)
                valid_meta = rest_metadata.sample(n=nval)
                train_meta = rest_metadata[~rest_metadata['newID'].isin(valid_meta['newID'])].groupby('Label').sample(n=spc,
                                                                                                                      replace=True)
                objects = train_meta['Class'].value_counts().reset_index()
                objects.to_csv(os.path.join(fold_path, 'objects.csv'), index=False)
            else:
                nval = int(opt.val_frac*rest_metadata.shape[0])
                valid_meta = rest_metadata.sample(n=nval)
                train_meta = rest_metadata[~rest_metadata['newID'].isin(valid_meta['newID'])]
                
            print('train: {} - val: {} test: {}'.format(train_meta.shape[0], valid_meta.shape[0], test_metadata.shape[0]))
    
            train_meta    = train_meta.copy()
            valid_meta    = valid_meta.copy()
            test_metadata = test_metadata.copy()
            
            train_meta.loc[:, 'subset_{}'.format(nfold)]    = ['train']*train_meta.shape[0]
            valid_meta.loc[:, 'subset_{}'.format(nfold)]    = ['validation']*valid_meta.shape[0]
            test_metadata.loc[:, 'subset_{}'.format(nfold)] = ['test']*test_metadata.shape[0]
    
            curr_meta = pd.concat([train_meta, valid_meta, test_metadata], axis=0)
            cols_to_use = curr_meta.columns.difference(metadata.columns)
            metadata = pd.merge(metadata, curr_meta[cols_to_use], left_index=True, right_index=True, how='outer')
    
        # ============== Pipeline ==============
        # ======================================
        if spc == '': spc = 'all'
        target_path = os.path.join(opt.target, str(spc))
        os.makedirs(target_path, exist_ok=True)
    
        metadata.to_parquet(os.path.join(target_path, 'metadata.parquet'), index=False)
        
        config['target']['path']              = target_path
        config['context_features']['path']    = os.path.join(opt.target, str(spc), 'metadata.parquet')
        config['sequential_features']['path'] = lcspath
    
        configPath = os.path.join(target_path, 'config.toml')
        with open(configPath, 'w') as file:
            toml.dump(config, file)
    
        pipeline = DataPipeline(metadata=metadata,
                                config_path=configPath)
        
        var = pipeline.run(n_jobs=opt.njobs,
                           elements_per_shard=opt.elements_per_shard)

        end = time.time()
        print('\n [INFO] ELAPSED: ', end - start)


if __name__ == '__main__':
    # python -m presentation.scripts.create_records --config ./data/my_data_folder/config.toml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/raw_parquet/alcock/', type=str,
                    help='Raw Data in parquet format to be used')
    parser.add_argument('--target', default='./data/records/alcock/', type=str,
                    help='Target folder where store the records')

    parser.add_argument('--ref-config', default='./data/config.toml', type=str,
                    help='Reference config file to be used as a template')

    parser.add_argument('--folds', default=1, type=int,
                    help='number of folds')
    
    parser.add_argument('--val-frac', default=0.2, type=float,
                    help='Fraction of the total training samples to use as validation')
    parser.add_argument('--ntest', default=100, type=float,
                    help='Number of samples per class for testing')

    parser.add_argument('--njobs', default=4, type=int,
                    help='Number of cores to use')
    
    parser.add_argument('--elements-per-shard', default=200000, type=int,
                    help='Number of light curves per shard')


    opt = parser.parse_args()        
    run(opt)