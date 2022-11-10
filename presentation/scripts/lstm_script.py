#!/usr/bin/python
import tensorflow as tf
import optuna
import pandas as pd
import os, sys
import joblib
from tqdm import tqdm

from core.data import pretraining_records
from presentation.scripts.classify import build_lstm
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop

GPU = sys.argv[1]
FOLD = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"]= GPU

def objective(lr, fold_n):
    data = './data/records/alcock/fold_{}/alcock_500'.format(fold_n)
    batch_size = 512
    max_obs = 200
    epochs=1000
    
    num_cls = pd.read_csv(os.path.join(data, 'objects.csv')).shape[0]

    train_batches = pretraining_records(os.path.join(data, 'train'),
                                        batch_size, max_obs=max_obs,
                                        msk_frac=0., rnd_frac=0., same_frac=0.,
                                        sampling=False, shuffle=True,
                                        n_classes=num_cls)

    val_batches = pretraining_records(os.path.join(data, 'val'),
                                      batch_size, max_obs=max_obs,
                                      msk_frac=0., rnd_frac=0., same_frac=0.,
                                      sampling=False, shuffle=False,
                                      n_classes=num_cls)
        
    model = build_lstm(maxlen=max_obs, n_classes=num_cls)
    
    optimizer = Adam(learning_rate=lr)
    
    model.compile(optimizer=optimizer,
              loss=CategoricalCrossentropy(from_logits=True),
              metrics='accuracy')
    
    target_dir = './runs/lstm_hp/fold_{}/{}'.format(fold_n, lr)
    
    estop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=20,
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

    _ = model.fit(train_batches,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[estop, tb],
                  validation_data=val_batches)
       
for lr in tqdm([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], total=5):
    objective(lr, FOLD)