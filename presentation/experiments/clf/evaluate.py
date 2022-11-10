#!/usr/bin/python
import pandas as pd
import subprocess
import pickle
import shutil
import os, sys
import sys
import time
import json

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, LSTM, LayerNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from core.data import load_dataset, inference_pipeline
from presentation.experiments.clf.classifiers import build_lstm, \
                                                     build_lstm_att, \
                                                     build_mlp_att
from core.astromer import ASTROMER


os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
ds_name = sys.argv[2]
case = sys.argv[3]
exp_mode = sys.argv[4]

if case == 'c':
    project_dir = './presentation/experiments/clf/finetuning/{}/c/{}/'.format(exp_mode, ds_name)
else:
    project_dir = './presentation/experiments/clf/finetuning/{}/ab/{}/'.format(exp_mode, ds_name)
datasets = ['{}_20'.format(ds_name),
            '{}_50'.format(ds_name),
            '{}_100'.format(ds_name),
            '{}_500'.format(ds_name)] 
exp_name    = './presentation/experiments/clf/classifiers/{}/{}/{}/'.format(exp_mode, case, ds_name)
data_path   = './data/records/{}/'.format(ds_name)


max_obs = 200
batch_size = 512
print('BATCH_SIZE: ',batch_size)


for model_arch in ['mlp_att', 'lstm', 'lstm_att']:
    print(model_arch)
    for ds in datasets:
        for fold_n in range(3):            
            astroweights = '{}/fold_{}/{}'.format(project_dir, fold_n, ds)
            print(astroweights)
            ds_path = '{}/fold_{}/{}'.format(data_path, fold_n, ds)
            target_dir = '{}/fold_{}/{}/{}'.format(exp_name, fold_n, ds, model_arch)
            
            if model_arch == 'lstm' and case != 'a':
                partial = './presentation/experiments/clf/classifiers/{}/a/{}/fold_{}/{}/{}'.format(exp_mode,
                                                                                                    ds_name, 
                                                                                                     fold_n, 
                                                                                                     ds, 
                                                                                                     model_arch)
                src = os.path.join('{}/results.pkl'.format(partial))
                os.makedirs(target_dir, exist_ok=True)
                dst = os.path.join(target_dir, 'results.pkl')
                shutil.copyfile(src, dst)
                continue
                
            n_classes = pd.read_csv(os.path.join(ds_path, 'objects.csv')).shape[0]
            dataset = load_dataset(os.path.join(ds_path, 'test'),repeat=1)
            test_batches = inference_pipeline(dataset,
                                              batch_size=batch_size,
                                              max_obs=max_obs,
                                              n_classes=n_classes,
                                              shuffle=False,
                                              get_ids=True)

            if model_arch == 'mlp_att':
                astromer = ASTROMER()
                astromer.load_weights(astroweights)
                model = build_mlp_att(astromer, max_obs, n_classes=n_classes,
                                      train_astromer=False)

            if model_arch == 'lstm_att':
                astromer = ASTROMER()
                astromer.load_weights(astroweights)
                model = build_lstm_att(astromer, max_obs, n_classes=n_classes,
                                       train_astromer=False)

            if model_arch == 'lstm':
                model = build_lstm(max_obs, n_classes=n_classes,
                                   state_dim=296)
            
            model.load_weights(os.path.join(target_dir, 'weights'))
            
            os.makedirs(target_dir, exist_ok=True)
            model.compile(optimizer='adam',
                          loss=CategoricalCrossentropy(from_logits=True),
                          metrics='accuracy')

            estop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=40,
                                  verbose=0,
                                  mode='auto',
                                  baseline=None,
                                  restore_best_weights=True)
            tb = TensorBoard(log_dir=os.path.join(target_dir, 'logs'),
                             write_graph=False,
                             write_images=False,
                             update_freq='epoch',
                             profile_batch=0,
                             embeddings_freq=0,
                             embeddings_metadata=None)


            y_pred = model.predict(test_batches)
            y_pred = tf.argmax(y_pred, 1)
            y_true = tf.concat([tf.argmax(y[0], 1) for _, y in test_batches], 0)
            y_oids = tf.concat([y[1] for _, y in test_batches], 0)
            
            out_dic = {'y_pred': y_pred.numpy(),
                       'y_true': y_true.numpy(),
                       'oid': y_oids.numpy()}
            with open(os.path.join(target_dir, 'results.pkl'), "wb") as output_file:
                pickle.dump(out_dic, output_file)

