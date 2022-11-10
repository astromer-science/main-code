import tensorflow as tf
import logging
import json
import os

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from core.astromer import get_ASTROMER
from core.metrics import custom_acc
from core.tboard import save_scalar
from core.losses import custom_bce
from core.output import SauceLayer
from tqdm import tqdm

from core.data import standardize
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def get_fc_attention(units, num_classes, weigths):
    ''' FC + ATT'''

    conf_file = os.path.join(weigths, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    model = get_ASTROMER(num_layers=conf['layers'],
                         d_model   =conf['head_dim'],
                         num_heads =conf['heads'],
                         dff       =conf['dff'],
                         base      =conf['base'],
                         dropout   =conf['dropout'],
                         maxlen    =conf['max_obs'],
                         use_leak  =conf['use_leak'])
    weights_path = '{}/weights'.format(weigths)
    model.load_weights(weights_path)
    encoder = model.get_layer('encoder')
    encoder.trainable = False

    mask = 1.-encoder.input['mask_in']
    x = encoder(encoder.input)
    x = x * mask
    x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)

    x = Dense(1024, name='FCN1')(x)
    x = Dense(512, name='FCN2')(x)
    x = Dense(num_classes, name='FCN3')(x)
    return Model(inputs=encoder.input, outputs=x, name="FCATT")

def get_lstm_no_attention(units, num_classes, maxlen, dropout=0.5):
    ''' LSTM + LSTM + FC'''

    serie  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='times')

    mask   = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='mask')
    length = Input(shape=(maxlen,),
                  batch_size=None,
                  dtype=tf.int32,
                  name='length')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times,
                   'length':length}

    bool_mask = tf.logical_not(tf.cast(placeholder['mask_in'], tf.bool))

    x = tf.concat([placeholder['times'], placeholder['input']], 2)

    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_0')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_1')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)

    return Model(inputs=placeholder, outputs=x, name="RNNCLF")

def get_lstm_attention(units, num_classes, weigths, dropout=0.5):
    ''' ATT + LSTM + LSTM + FC'''
    conf_file = os.path.join(weigths, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    model = get_ASTROMER(num_layers=conf['layers'],
                         d_model   =conf['head_dim'],
                         num_heads =conf['heads'],
                         dff       =conf['dff'],
                         base      =conf['base'],
                         dropout   =conf['dropout'],
                         maxlen    =conf['max_obs'],
                         use_leak  =conf['use_leak'])
    weights_path = '{}/weights'.format(weigths)
    model.load_weights(weights_path)
    encoder = model.get_layer('encoder')
    encoder.trainable = False

    bool_mask = tf.logical_not(tf.cast(encoder.input['mask_in'], tf.bool))

    x = encoder(encoder.input)
    x = tf.reshape(x, [-1, conf['max_obs'], encoder.output.shape[-1]])
    x = LayerNormalization()(x)

    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_0')(x, mask=bool_mask)
    x = LayerNormalization()(x)
    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_1')(x, mask=bool_mask)
    x = LayerNormalization()(x)
    x = Dense(num_classes, name='FCN')(x)

    return Model(inputs=encoder.input, outputs=x, name="RNNCLF")

@tf.function
def train_step(model, batch, opt):
    with tf.GradientTape() as tape:
        y_pred = model(batch)
        ce = custom_bce(y_true=batch['label'], y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    grads = tape.gradient(ce, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return acc, ce

@tf.function
def valid_step(model, batch, return_pred=False):
    with tf.GradientTape() as tape:
        y_pred = model(batch, training=False)
        ce = custom_bce(y_true=batch['label'],
                         y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    if return_pred:
        return acc, ce, y_pred, batch['label']
    return acc, ce

def train(model,
          train_batches,
          valid_batches,
          patience=20,
          exp_path='./experiments/test',
          epochs=1,
          lr=1e-3,
          verbose=1):
    # Tensorboard
    train_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'train'))
    valid_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'valid'))
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(lr)
    # To save metrics
    train_bce  = tf.keras.metrics.Mean(name='train_bce')
    valid_bce  = tf.keras.metrics.Mean(name='valid_bce')
    train_acc  = tf.keras.metrics.Mean(name='train_acc')
    valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

    # ==============================
    # ======= Training Loop ========
    # ==============================
    best_loss = 999999.
    es_count = 0
    pbar = tqdm(range(epochs), desc='epoch')
    for epoch in pbar:
        for train_batch in train_batches:
            acc, bce = train_step(model, train_batch, optimizer)
            train_acc.update_state(acc)
            train_bce.update_state(bce)

        for valid_batch in valid_batches:
            acc, bce = valid_step(model, valid_batch)
            valid_acc.update_state(acc)
            valid_bce.update_state(bce)

        save_scalar(train_writter, train_acc, epoch, name='accuracy')
        save_scalar(valid_writter, valid_acc, epoch, name='accuracy')
        save_scalar(train_writter, train_bce, epoch, name='xentropy')
        save_scalar(valid_writter, valid_bce, epoch, name='xentropy')

        if valid_bce.result() < best_loss:
            best_loss = valid_bce.result()
            es_count = 0.
            model.save_weights(os.path.join(exp_path, 'weights'))
        else:
            es_count+=1.
        if es_count == patience:
            print('[INFO] Early Stopping Triggered')
            break


        msg = 'EPOCH {} - ES COUNT: {}/{} Train acc: {:.4f} - Val acc: {:.4f} - Train CE: {:.2f} - Val CE: {:.2f}'.format(
                                                                                      epoch,
                                                                                      es_count,
                                                                                      patience,
                                                                                      train_acc.result(),
                                                                                      valid_acc.result(),
                                                                                      train_bce.result(),
                                                                                      valid_bce.result())

        pbar.set_description(msg)

        valid_bce.reset_states()
        train_bce.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()

def get_conf(path):
    conf_file = os.path.join(path, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)
        return conf

def load_weights(model, weigths):
    weights_path = '{}/weights'.format(weigths)
    model.load_weights(weights_path)
    return model

def predict(model, test_batches):
    predictions = []
    true_labels = []
    for batch in tqdm(test_batches, desc='test'):
        acc, ce, y_pred, y_true = valid_step(model, batch, return_pred=True)
        if len(y_pred.shape)>2:
            predictions.append(y_pred[:, -1, :])
        else:
            predictions.append(y_pred)

        true_labels.append(y_true)

    y_pred = tf.concat(predictions, 0)
    y_true = tf.concat(true_labels, 0)
    pred_labels = tf.argmax(y_pred, 1)

    precision, \
    recall, \
    f1, _ = precision_recall_fscore_support(y_true,
                                            pred_labels,
                                            average='macro')
    acc = accuracy_score(y_true, pred_labels)
    results = {'f1': f1,
               'recall': recall,
               'precision': precision,
               'accuracy':acc,
               'y_true':y_true,
               'y_pred':pred_labels}

    return results


def predict_from_path(path, test_batches, mode=0, save=False):
    conf_rnn = get_conf(path)

    if mode == 0:
        clf = get_lstm_attention(conf_rnn['units'],
                               conf_rnn['num_classes'],
                               conf_rnn['w'],
                                     conf_rnn['dropout'])
    if mode == 1:
        clf = get_fc_attention(conf_rnn['units'],
                                     conf_rnn['num_classes'],
                                     conf_rnn['w'])
    if mode == 2:
        clf = get_lstm_no_attention(conf_rnn['units'],
                                    conf_rnn['num_classes'],
                                    conf_rnn['max_obs'],
                                    conf_rnn['dropout'])

    clf = load_weights(clf, path)
    results = predict(clf, test_batches)

    return results
