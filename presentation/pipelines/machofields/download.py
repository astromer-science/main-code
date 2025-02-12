import pandas as pd
import argparse
import time
import toml
import os

from tqdm import tqdm 
from presentation.pipelines.machofields.utils import connect_to_drive

def run(opt):
    drive = connect_to_drive()

    start = time.time()
    
    os.makedirs(os.path.join(opt.temp_dir, 'light_curves'), exist_ok=True)


    file_list = drive.ListFile({'includeItemsFromAllDrives':True,
                            'driveId':'0ANj7Lpiu7QacUk9PVA',
                            'corpora':'drive',
                            'supportsAllDrives':True,
                            'q': "'{}' in parents and trashed=false".format(opt.id)}).GetList()

    pbar = tqdm(file_list, total=len(file_list))
    i = 0
    to_change = {}
    for file in pbar:    
        if 'metadata' in file['originalFilename']:
            target_file = os.path.join(opt.temp_dir, 'metadata.parquet')
        else:
            shard_name = str(i).rjust(3, '0')
            filename = 'shard_{}.parquet'.format(shard_name)
            target_file = os.path.join(opt.temp_dir, 'light_curves', filename)
            to_change[file['originalFilename']] = str(i)
            i+=1

        file.GetContentFile(target_file)
                       

    # ============================================================================
    # FIXING FORMAT ==============================================================
    metadata  = pd.read_parquet(os.path.join(opt.temp_dir, 'metadata.parquet'))
    new_meta = metadata.drop_duplicates('newID')
    del new_meta['Band']
    cardinality = metadata.groupby(['newID']).count()['Band'].reset_index()
    new_meta = pd.merge(new_meta, cardinality, on='newID', how='inner')
    new_meta['Label'] = pd.Categorical(new_meta['Class']).codes
    new_meta['shard'] = new_meta['shard'].replace(to_change)
    cond = new_meta['shard'].str.endswith('.parquet')
    new_meta = new_meta[~cond]

    new_meta['shard']  = new_meta['shard'].astype(int)
    new_meta['Band'] = new_meta['Band'].astype(int)
    new_meta['Class'] = new_meta['Class'].astype(str)
    new_meta.to_parquet(os.path.join(opt.temp_dir, 'metadata.parquet'))
    # ============================================================================
    # ============================================================================

    target_folder = "{}/{}".format(opt.target, opt.field)
    os.makedirs(target_folder, exist_ok=True)

    config = {
        "id_column": {
            "value": "newID",
            "dtype": "integer",
        },
        "target": {
            "path": target_folder,
            "dtype": "string"
        },
        "context_features": {
            'path': "./data/temp/cleaned_metadata.parquet", #USE FILTERED VERSION
            'dtype': "string",
            'value': [ "ID", "Class", "Band", "Label", "shard",],
            'dtypes': [ "string", "string", "integer", "integer", "integer",]
        },
        "sequential_features": {
            'path': "./data/temp/light_curves",
            'value': [ "observation_date", "mag", "err"],
            'dtypes': [ "float", "float", "float"],
        }
    }
    with open(os.path.join(target_folder, 'config.toml'), "w") as f:
        toml.dump(config, f)

    end = time.time()
    print('\n [INFO] ELAPSED: ', end - start)

if __name__ == '__main__':
    # python -m presentation.scripts.create_records --config ./data/my_data_folder/config.toml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='drive id', type=str,
                    help='Folder ID')
    parser.add_argument('--field', default='F_309', type=str,
                    help='MACHO Field ID')
    parser.add_argument('--temp-dir', default='./data/temp', type=str,
                    help='Temporal directory where save raw data to create records')
    parser.add_argument('--target', default='./data/records/bigmacho', type=str,
                    help='Target directory where records will be stored')

    opt = parser.parse_args()        
    run(opt)
