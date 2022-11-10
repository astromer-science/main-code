import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import json
import time
import h5py
import os

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, LSTM, LayerNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop

from core.data  import pretraining_records
from core.astromer import get_ASTROMER

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings\

def normalize_batch(tensor):
    min_ = tf.expand_dims(tf.reduce_min(tensor, 1), 1)
    max_ = tf.expand_dims(tf.reduce_max(tensor, 1), 1)
    tensor = tf.math.divide_no_nan(tensor - min_, max_ - min_)
    return tensor

class NormedLSTMCell(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = ((self.units, self.units), (self.units, self.units))

        super(NormedLSTMCell, self).__init__(**kwargs)

        self.cell_0 = tf.keras.layers.LSTMCell(self.units)
        self.cell_1 = tf.keras.layers.LSTMCell(self.units)
        self.bn = LayerNormalization(name='bn_step')

    def call(self, inputs, states, training=False):
        s0, s1 = states[0], states[1]
        output, s0 = self.cell_0(inputs, states=s0, training=training)
        output = self.bn(output, training=training)
        output, s1 = self.cell_1(output, states=s1, training=training)
        return output, [s0, s1]

    def get_config(self):
        config = super(NormedLSTMCell, self).get_config().copy()
        config.update({"units": self.units})
        return config


def build_lstm(maxlen, n_classes):
    print('[INFO] Building LSTM Baseline')
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    m = tf.cast(1.-placeholder['mask_in'][...,0], tf.bool)
    tim = normalize_batch(placeholder['times'])
    inp = normalize_batch(placeholder['input'])
    x = tf.concat([tim, inp], 2)

    cell_0 = NormedLSTMCell(units=256)
    dense  = Dense(n_classes, name='FCN')

    s0 = [tf.zeros([tf.shape(x)[0], 256]),
          tf.zeros([tf.shape(x)[0], 256])]
    s1 = [tf.zeros([tf.shape(x)[0], 256]),
          tf.zeros([tf.shape(x)[0], 256])]

    rnn = tf.keras.layers.RNN(cell_0, return_sequences=False)
    x = rnn(x, initial_state=[s0, s1], mask=m)
    x = tf.nn.dropout(x, .3)
    x = dense(x)
    return Model(placeholder, outputs=x, name="LSTM")

def build_lstm_att(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')
    print('BUILDING NEW LSTM + ATT')
    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    cell_0 = NormedLSTMCell(units=256)
    dense  = Dense(n_classes, name='FCN')

    s0 = [tf.zeros([tf.shape(placeholder['input'])[0], 256]),
          tf.zeros([tf.shape(placeholder['input'])[0], 256])]
    s1 = [tf.zeros([tf.shape(placeholder['input'])[0], 256]),
          tf.zeros([tf.shape(placeholder['input'])[0], 256])]
    rnn = tf.keras.layers.RNN(cell_0, return_sequences=False)

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer

    mask = tf.cast(1.-placeholder['mask_in'][...,0], dtype=tf.bool)
    x = encoder(placeholder, training=train_astromer)
    x = tf.math.divide_no_nan(x-tf.expand_dims(tf.reduce_mean(x, 1),1),
                              tf.expand_dims(tf.math.reduce_std(x, 1), 1))
    x = rnn(x, initial_state=[s0, s1], mask=mask)
    x = tf.nn.dropout(x, .3)
    x = dense(x)
    return Model(placeholder, outputs=x, name="LSTM_ATT")

def build_mlp_att(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer

    mask = 1.-placeholder['mask_in']

    x = encoder(placeholder, training=False)
    x = x * mask
    x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="FCATT")

def build_mini_mlp_att(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer

    mask = 1.-placeholder['mask_in']

    x = encoder(placeholder, training=False)
    x = x * mask
    x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="MLP_ATT_MINI")

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
    # Loading data

    num_cls = pd.read_csv(os.path.join(opt.data, 'objects.csv')).shape[0]

    train_batches = pretraining_records(os.path.join(opt.data, 'train'),
                                        opt.batch_size, max_obs=opt.max_obs,
                                        msk_frac=0., rnd_frac=0., same_frac=0.,
                                        sampling=False, shuffle=True,
                                        n_classes=num_cls)

    val_batches = pretraining_records(os.path.join(opt.data, 'val'),
                                      opt.batch_size, max_obs=opt.max_obs,
                                      msk_frac=0., rnd_frac=0., same_frac=0.,
                                      sampling=False, shuffle=False,
                                      n_classes=num_cls)

    conf_file = os.path.join(opt.w, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    astromer = get_ASTROMER(num_layers=conf['layers'],
                         d_model   =conf['head_dim'],
                         num_heads =conf['heads'],
                         dff       =conf['dff'],
                         base      =conf['base'],
                         dropout   =conf['dropout'],
                         maxlen    =conf['max_obs'],
                         use_leak  =conf['use_leak'])

    weights_path = '{}/weights'.format(opt.w)
    astromer.load_weights(weights_path)
    print('[INFO] Data: {}'.format(opt.data))
    print('[INFO] ASTROMER weights: {}'.format(weights_path))
    print('[INFO] Training astromer? ', opt.finetune)

    if opt.mode == 'lstm_att':
        model = build_lstm_att(astromer,
                               maxlen=opt.max_obs,
                               n_classes=num_cls,
                               train_astromer=opt.finetune)
    if opt.mode == 'mini_mlp_att':
        model = build_mini_mlp_att(astromer,
                                   maxlen=opt.max_obs,
                                   n_classes=num_cls,
                                   train_astromer=True)

    if opt.mode == 'mlp_att':
        model = build_mlp_att(astromer,
                              maxlen=opt.max_obs,
                              n_classes=num_cls,
                              train_astromer=opt.finetune)
    if opt.mode == 'lstm':
        model = build_lstm(maxlen=opt.max_obs,
                           n_classes=num_cls)
    print('[INFO] {} created '.format(opt.mode))

    target_dir = os.path.join(opt.p, '{}_2'.format(opt.mode))
    optimizer = Adam(learning_rate=opt.lr)

    os.makedirs(target_dir, exist_ok=True)

    model.compile(optimizer=optimizer,
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')

    estop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=opt.patience,
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
                  epochs=opt.epochs,
                  batch_size=opt.batch_size,
                  callbacks=[estop, tb],
                  validation_data=val_batches)

    model.save(os.path.join(target_dir, 'model'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')
    # ASTROMER
    parser.add_argument('--w', default="./runs/huge", type=str,
                        help='ASTROMER pretrained weights')
    parser.add_argument('--finetune', action='store_true', default=False)
    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/alcock', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')

    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=20, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')

    # RNN HIPERPARAMETERS
    parser.add_argument('--mode', default='lstm', type=str,
                        help='Classifier lstm, lstm_att, or mlp_att')

    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU number to be used')

    opt = parser.parse_args()
    run(opt)
