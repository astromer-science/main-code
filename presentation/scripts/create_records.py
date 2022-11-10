import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import sys,os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from core.masking import get_padding_mask
from core.astromer import get_ASTROMER, train
from core.data  import (create_dataset, 
                        pretraining_records)


source = './data/raw_data/atlas/LCs/filter_o/' # lightcurves folder
metadata = './data/raw_data/atlas/metadata_o.dat' # metadata file
name = 'big_atlas'

meta = pd.read_csv(metadata)
meta['Band'] = tf.ones(meta.shape[0])
meta = meta.rename(columns={'objID':'ID', 'Unnamed: 0':'ID', 'Path_R':'Path'})
meta['Path'] = meta['ID'].astype(str)+'.dat'

for fold_n in range(1):
    target = './data/records/{}/fold_{}/{}'.format(name, fold_n, name)
    
    test_meta  = meta.sample(frac=0.2)
    train_meta = meta[~meta['ID'].isin(test_meta['ID'])]

    create_dataset(train_meta, source, target, max_lcs_per_record=20000, 
                   n_jobs=7, subsets_frac=(0.8, 0.2), test_subset=test_meta,
                   sep=';')