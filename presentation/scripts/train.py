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
    # Get model
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    astromer = get_ASTROMER(num_layers=opt.layers,
                            d_model=opt.head_dim,
                            num_heads=opt.heads,
                            dff=opt.dff,
                            base=opt.base,
                            dropout=opt.dropout,
                            maxlen=opt.max_obs,
                            use_leak=opt.use_leak,
                            no_train=opt.no_train)

    # Make sure we don't overwrite a previous training
    opt.p = get_folder_name(opt.p, prefix='')

    # Creating (--p)royect directory
    os.makedirs(opt.p, exist_ok=True)

    # Save Hyperparameters
    conf_file = os.path.join(opt.p, 'conf.json')
    varsdic = vars(opt)
    varsdic['exp_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    with open(conf_file, 'w') as json_file:
        json.dump(varsdic, json_file, indent=4)

    # Loading data
    train_batches = pretraining_records(os.path.join(opt.data, 'train'),
                                        opt.batch_size,
                                        max_obs=opt.max_obs,
                                        shuffle=True,
                                        sampling=True,
                                        repeat=opt.repeat,
                                        msk_frac=opt.msk_frac,
                                        rnd_frac=opt.rnd_frac,
                                        same_frac=opt.same_frac)
    valid_batches = pretraining_records(os.path.join(opt.data, 'val'),
                                        opt.batch_size,
                                        max_obs=opt.max_obs,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=opt.msk_frac,
                                        rnd_frac=opt.rnd_frac,
                                        same_frac=opt.same_frac)

    
    # Training ASTROMER
    train(astromer, train_batches, valid_batches,
          patience=opt.patience,
          exp_path=opt.p,
          epochs=opt.epochs,
          lr=opt.lr,
          verbose=0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')
    parser.add_argument('--repeat', default=1, type=int,
                    help='Number of times for sampling windows from single LC')
    parser.add_argument('--msk-frac', default=0.5, type=float,
                        help='[MASKED] fraction')
    parser.add_argument('--rnd-frac', default=0.2, type=float,
                        help='Fraction of [MASKED] to be replaced by random values')
    parser.add_argument('--same-frac', default=0.2, type=float,
                        help='Fraction of [MASKED] to be replaced by same values')

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=40, type=int,
                        help='batch size')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU to use')

    # ASTROMER HIPERPARAMETERS
    parser.add_argument('--layers', default=2, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--heads', default=4, type=int,
                        help='Number of self-attention heads')
    parser.add_argument('--head-dim', default=256, type=int,
                        help='Head-attention Dimensionality ')
    parser.add_argument('--dff', default=128, type=int,
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
    parser.add_argument('--no-shuffle', default=False, action='store_true',
                        help='No shuffle training and validation set')

    opt = parser.parse_args()
    run(opt)
