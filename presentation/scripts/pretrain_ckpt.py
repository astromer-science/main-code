import tensorflow as tf
import argparse
import math
import toml
import os

from tensorflow.keras.optimizers import Adam
from datetime import datetime
from tqdm import tqdm

from src.training.scheduler import CustomSchedule
from src.training.utils import train_step, test_step, tensorboard_log

from presentation.pipelines.steps.model_design import build_model, load_pt_model
from presentation.pipelines.steps.load_data import build_loader
from presentation.pipelines.steps.metrics import evaluate_ft


def replace_config(source, target):
    for key in ['data', 'no_cache', 'exp_name', 'checkpoisnt', 
                'gpu', 'lr', 'bs', 'patience', 'num_epochs', 'scheduler']:
        target[key] = source[key]
    return target

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    ROOT = './presentation/'
    trial = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPDIR = os.path.join(ROOT, 'results', opt.exp_name, trial, 'pretraining')
    print('[INFO] Saving weights on {}'.format(EXPDIR))
    os.makedirs(EXPDIR, exist_ok=True)
    # ======= MODEL ========================================
    if opt.checkpoint != '-1':
        print('[INFO] Restoring previous training')
        astromer, pconfig = load_pt_model(opt.checkpoint, optimizer=None)
        opt.__dict__ = replace_config(source=opt.__dict__, target=pconfig)
    else:
        astromer = build_model(opt.__dict__)
        
    # ========== DATA ========================================
    loaders = build_loader(data_path=opt.data, 
                           params=opt.__dict__,
                           batch_size=opt.bs,
                           debug=opt.debug,
                           normalize=opt.norm,
                           sampling=opt.sampling,
                           repeat=opt.repeat,
                           return_test=True,
                           )
    # ========== COMPILE =====================================
    if opt.scheduler:
        print('[INFO] Using Custom Scheduler')
        lr = CustomSchedule(d_model=int(opt.head_dim*opt.num_heads))
    else:
        lr = opt.lr

    optimizer = Adam(lr, 
                     beta_1=0.9,
                     beta_2=0.98,
                     epsilon=1e-9,
                     name='astromer_optimizer')

    with open(os.path.join(EXPDIR, 'config.toml'), 'w') as f:
        toml.dump(opt.__dict__, f)

    
    # TRAINING LOOP
    train_writer = tf.summary.create_file_writer(os.path.join(EXPDIR, 'tensorboard', 'train'))
    valid_writer = tf.summary.create_file_writer(os.path.join(EXPDIR, 'tensorboard', 'validation'))

    pbar  = tqdm(range(opt.num_epochs), total=opt.num_epochs)
    pbar.set_description("Epoch 0 (p={}) - rmse: -/- rsquare: -/-", refresh=True)
    pbar.set_postfix(item=0)    

    # ========= Training Loop ==================================
    es_count = 0
    min_loss = 1e9
    for epoch in pbar:
        pbar.set_postfix(item1=epoch)
        epoch_tr_rmse    = []
        epoch_tr_rsquare = []
        epoch_tr_loss    = []
        epoch_vl_rmse    = []
        epoch_vl_rsquare = []
        epoch_vl_loss    = []

        for numbatch, batch in enumerate(loaders['train']):
            pbar.set_postfix(item=numbatch)
            metrics = train_step(astromer, batch, optimizer)
            epoch_tr_rmse.append(metrics['rmse'])
            epoch_tr_rsquare.append(metrics['rsquare'])
            epoch_tr_loss.append(metrics['loss'])

        for batch in loaders['validation']:
            metrics = test_step(astromer, batch)
            epoch_vl_rmse.append(metrics['rmse'])
            epoch_vl_rsquare.append(metrics['rsquare'])
            epoch_vl_loss.append(metrics['loss'])

        tr_rmse    = tf.reduce_mean(epoch_tr_rmse)
        tr_rsquare = tf.reduce_mean(epoch_tr_rsquare)
        vl_rmse    = tf.reduce_mean(epoch_vl_rmse)
        vl_rsquare = tf.reduce_mean(epoch_vl_rsquare)
        tr_loss    = tf.reduce_mean(epoch_tr_loss)
        vl_loss    = tf.reduce_mean(epoch_vl_loss)

        tensorboard_log('loss', tr_loss, train_writer, step=epoch)
        tensorboard_log('loss', vl_loss, valid_writer, step=epoch)
        
        tensorboard_log('rmse', tr_rmse, train_writer, step=epoch)
        tensorboard_log('rmse', vl_rmse, valid_writer, step=epoch)
        
        tensorboard_log('rsquare', tr_rsquare, train_writer, step=epoch)
        tensorboard_log('rsquare', vl_rsquare, valid_writer, step=epoch)
        
        if epoch % 50 == 0:
            print('[INFO] Saving chekcpoints at epoch {}'.format(epoch))
            astromer.save_weights(os.path.join(EXPDIR, 'ckpts', 'epoch_{}'.format(epoch)))

        if tf.math.greater(min_loss, vl_rmse):
            min_loss = vl_rmse
            es_count = 0
            astromer.save_weights(os.path.join(EXPDIR, 'ckpts', 'best_{}'.format(epoch)))
        else:
            es_count = es_count + 1

        if es_count == opt.patience:
            print('[INFO] Early Stopping Triggered at epoch {:03d}'.format(epoch))
            break
        
        pbar.set_description("Epoch {} (p={}) - rmse: {:.3f}/{:.3f} rsquare: {:.3f}/{:.3f}".format(epoch, 
                                                                                            es_count,
                                                                                            tr_rmse,
                                                                                            vl_rmse,
                                                                                            tr_rsquare,
                                                                                            vl_rsquare))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==== ECOSYSTEM ==================================================
    parser.add_argument('--exp-name', default='alcock_test', type=str,
                    help='Project name')    
    parser.add_argument('--checkpoint', default='-1', type=str,
                        help='Restore training by using checkpoints. This is the route to the checkpoint folder.')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


    # ==== DATA =======================================================
    parser.add_argument('--data', default='./data/records/alcock/fold_0/alcock', type=str,
                    help='Data folder where tf.record files are located')
    parser.add_argument('--repeat', default=1, type=int,
                        help='repeat data')
    parser.add_argument('--window-size', default=200, type=int,
                        help='windows size of the PSFs')
    parser.add_argument('--no-cache', action='store_true', help='no cache dataset')
    parser.add_argument('--probed', default=0.5, type=float,
                        help='Probed percentage')
    parser.add_argument('--rs', default=0.2, type=float,
                        help='Probed fraction to be randomized or unmasked')
    parser.add_argument('--same', default=0.2, type=float,
                        help='Fraction to make visible during masked-self attention while evaluating during loss')
    parser.add_argument('--norm', default='zero-mean', type=str,
                        help='normalization: zero-mean - random-mean')
    parser.add_argument('--sampling', action='store_true', help='sampling windows')
    parser.add_argument('--no-msk-token', action='store_true', help='Do not add trainable MSK token in the input')

    # ==== TRAINING ===================================================
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--bs', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--patience', default=20, type=int,
                        help='Earlystopping threshold in number of epochs')
    parser.add_argument('--num_epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--scheduler', action='store_true', help='Use Custom Scheduler during training')
    parser.add_argument('--correct-loss', action='store_true', help='Use error bars to weigh loss')

    # ==== MODEL ======================================================
    parser.add_argument('--arch', default='base', type=str,
                        help='Astromer architecture: "zero" (paper) or "base"(new version)')
    parser.add_argument('--num-layers', default=2, type=int,
                        help='Number of Attention Layers')
    parser.add_argument('--num-heads', default=4, type=int,
                        help='Number of heads within the attention layer')
    parser.add_argument('--head-dim', default=64, type=int,
                        help='Head dimension')
    parser.add_argument('--pe-dim', default=256, type=int,
                        help='Positional encoder size - i.e., Number of frequencies')
    parser.add_argument('--pe-base', default=10000, type=int,
                        help='Positional encoder base')
    parser.add_argument('--pe-exp', default=2, type=int,
                        help='Positional encoder exponent')
    parser.add_argument('--mixer', default=128, type=int,
                        help='Units to be used on the hidden layer of a feed-forward network that combines head outputs within an attention layer')
    parser.add_argument('--dropout', default=0., type=float,
                        help='Dropout to use on the output of each attention layer (before mixer layer)')
    parser.add_argument('--m-alpha', default=-1000000000, type=float,
                        help='Alpha used within mask self-attention. -1e9 by default. Use 1 for "zero" arch')
    parser.add_argument('--mask-format', default='K', type=str,
                        help='mask on Query and Key tokens (QK) or Query tokens only (Q)')
    parser.add_argument('--loss-format', default='rmse', type=str,
                        help='what consider during loss: rmse - mse - p')
    parser.add_argument('--use-leak', action='store_true',
                        help='Use Custom Scheduler during training')  
    parser.add_argument('--temperature', default=0., type=float,
                        help='Temperature used within the softmax argument')





    opt = parser.parse_args()        
    run(opt)
