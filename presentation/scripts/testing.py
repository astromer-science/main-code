import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
import json
import time
import h5py
import os

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop

from core.data  import pretraining_records
from core.utils import get_metrics
from core.astromer import get_ASTROMER, predict
from presentation.scripts.classify import build_lstm, build_lstm_att, build_mlp_att
from sklearn.metrics import precision_recall_fscore_support

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings\


def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
    # =======================
    # TESTING FINETUNING
    # =======================
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
 
    print('[INFO] ASTROMER weights: {}'.format(weights_path))

    # =======================
    # TESTING CLASSIFICATION
    # =======================
    num_cls = pd.read_csv(os.path.join(opt.data, 'objects.csv')).shape[0]
    test_batches = pretraining_records(os.path.join(opt.data, 'test'),
                                      opt.batch_size, max_obs=opt.max_obs,
                                      msk_frac=0., rnd_frac=0., same_frac=0.,
                                      sampling=False, shuffle=False,
                                      n_classes=num_cls)
    
    if opt.mode == 'lstm_att':
        model = build_lstm_att(astromer,
                               maxlen=conf['max_obs'],
                               n_classes=num_cls,
                               train_astromer=False)
    if opt.mode == 'mlp_att':
        model = build_mlp_att(astromer,
                              maxlen=conf['max_obs'],
                              n_classes=num_cls,
                              train_astromer=False)
    if opt.mode == 'lstm':
        model = build_lstm(maxlen=conf['max_obs'],
                           n_classes=num_cls)
    print('[INFO] {} created '.format(opt.mode))
        
    target_dir = os.path.join(opt.p, '{}_2'.format(opt.mode))

    model.compile(optimizer=Adam(learning_rate=opt.lr),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')
    model.load_weights(os.path.join(target_dir, 'model'))  
  
    clf_metrics = get_metrics(os.path.join(target_dir, 'logs', 'validation'))

    # Training time in seconds
    dt = (clf_metrics['wall_time'].values[-1] - clf_metrics['wall_time'].values[0])
    
    y_pred = model.predict(test_batches)
    y_pred = tf.argmax(y_pred, 1)
    
    y_true = tf.concat([tf.argmax(y, 1) for _, y in test_batches], 0)
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    # ======================
    # ==== SAVE RESULTS ====
    # ======================
    clf_metrics = {'prec': prec, 'rec': rec, 'f1': f1, 'time':dt, 'y_pred':y_pred, 'y_true':y_true}
    
    out_dic = {
        'classification':clf_metrics
    }
    with open(os.path.join(target_dir, 'summary.pkl'), "wb") as output_file:
        pickle.dump(out_dic, output_file)
    
    
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

    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU number to be used')

    opt = parser.parse_args()
    run(opt)
