import tensorflow as tf
import argparse
import logging
import json
import time
import os

from core.astromer import get_ASTROMER, train
from core.data  import pretraining_records
from core.utils import get_folder_name
from time import gmtime, strftime

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings



def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

    # Get model
    astromer = get_ASTROMER(num_layers=opt.layers,
                            d_model=opt.head_dim,
                            num_heads=opt.heads,
                            dff=opt.dff,
                            base=opt.base,
                            dropout=opt.dropout,
                            maxlen=opt.max_obs,
                            use_leak=opt.use_leak,
                            no_train=opt.no_train)

    # Check for pretrained weigths
    if os.path.isfile(os.path.join(opt.w, 'weights.h5')):
        os.makedirs(opt.p, exist_ok=True)

        print('[INFO] Pretrained model detected! - Finetuning...')
        conf_file = os.path.join(opt.w, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)
        # Loading hyperparameters of the pretrained model
        astromer = get_ASTROMER(num_layers=conf['layers'],
                                d_model=conf['head_dim'],
                                num_heads=conf['heads'],
                                dff=conf['dff'],
                                base=conf['base'],
                                dropout=conf['dropout'],
                                maxlen=conf['max_obs'],
                                use_leak=conf['use_leak'],
                                no_train=conf['no_train'])

        # Loading pretrained weights
        weights_path = '{}/weights.h5'.format(opt.w)
        astromer.load_weights(weights_path)
        # Defining a new ()--p)roject folder



        # Save Hyperparameters
        conf_file = os.path.join(opt.p, 'conf.json')
        varsdic = vars(opt)

        for key in conf.keys():
            if key in ['batch_size', 'p', 'repeat', 'data', 'patience']:
                continue
            varsdic[key] = conf[key]

        varsdic['exp_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        with open(conf_file, 'w') as json_file:
            json.dump(varsdic, json_file, indent=4)

        # Loading data
        train_batches = pretraining_records(os.path.join(opt.data, 'train'),
                                            opt.batch_size,
                                            max_obs=conf['max_obs'],
                                            msk_frac=conf['msk_frac'],
                                            rnd_frac=conf['rnd_frac'],
                                            same_frac=conf['same_frac'],
                                            sampling=False,
                                            shuffle=True)

        valid_batches = pretraining_records(os.path.join(opt.data, 'val'),
                                            opt.batch_size,
                                            max_obs=conf['max_obs'],
                                            msk_frac=conf['msk_frac'],
                                            rnd_frac=conf['rnd_frac'],
                                            same_frac=conf['same_frac'],
                                            sampling=False,
                                            shuffle=True)


        # Training ASTROMER
        train(astromer, train_batches, valid_batches,
              patience=opt.patience,
              exp_path=opt.p,
              epochs=opt.epochs,
              verbose=0,
              lr=opt.lr)
    else:
        print('[ERROR] No weights found to finetune')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--w', default="./weights/astromer_10022021", type=str,
                        help='astromer weigths')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=40, type=int,
                        help='batch size')

    # ASTROMER HIPERPARAMETERS
    parser.add_argument('--layers', default=1, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--heads', default=2, type=int,
                        help='Number of self-attention heads')
    parser.add_argument('--head-dim', default=256, type=int,
                        help='Head-attention Dimensionality ')
    parser.add_argument('--dff', default=512, type=int,
                        help='Dimensionality of the middle  dense layer at the end of the encoder')
    parser.add_argument('--dropout', default=0.1 , type=float,
                        help='dropout_rate for the encoder')
    parser.add_argument('--base', default=1000, type=int,
                        help='base of embedding')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')

    parser.add_argument('--use-leak', default=False, action='store_true',
                        help='Add the input to the attention vector')
    parser.add_argument('--no-train', default=False, action='store_true',
                        help='Train self-attention layer')
    parser.add_argument('--shuffle', default=False, action='store_true',
                        help='Shuffle while training datasets')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU to use')
    opt = parser.parse_args()
    run(opt)
