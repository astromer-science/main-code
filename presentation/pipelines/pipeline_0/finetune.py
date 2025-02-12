import tensorflow as tf
import argparse
import toml
import sys
import os

from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from tensorflow.keras.optimizers import Adam
from presentation.pipelines.steps.model_design import load_pt_model 
from presentation.pipelines.steps.load_data import build_loader 
from presentation.pipelines.steps.metrics import evaluate_ft
from src.training.utils import train
from presentation.scripts.disttrain import distributed_train_step, distributed_test_step, tensorboard_log
from tqdm import tqdm



def ft_step(opt):
    factos = opt.data.split('/')
    ft_model = '/'.join(factos[-3:])

    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    devices = ['/gpu:{}'.format(dev) for dev in opt.gpu.split(',')]
    mirrored_strategy = tf.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    
    EXPDIR = os.path.join(opt.pt_model, '..', opt.exp_name, ft_model)
    print('[INFO] Exp dir: ', EXPDIR)

    os.makedirs(EXPDIR, exist_ok=True)  
    
    config_file = os.path.join(opt.pt_model, 'config.toml')
    with open(config_file, 'r') as file:
        pt_config = toml.load(file)
    # ========== DATA ========================================
    loaders = build_loader(opt.data, 
                           pt_config, 
                           batch_size=opt.bs, 
                           clf_mode=False, 
                           normalize='zero-mean', 
                           sampling=False,
                           repeat=1,
                           return_test=True)
    
    train_batches = mirrored_strategy.experimental_distribute_dataset(loaders['train'])
    valid_batches = mirrored_strategy.experimental_distribute_dataset(loaders['validation'])
    train_writer = tf.summary.create_file_writer(os.path.join(EXPDIR, 'tensorboard', 'train'))
    valid_writer = tf.summary.create_file_writer(os.path.join(EXPDIR, 'tensorboard', 'validation'))
                            
    with mirrored_strategy.scope():
        
        astromer, model_config = load_pt_model(opt.pt_model)
            
        optimizer = Adam(opt.lr, 
                         beta_1=0.9,
                         beta_2=0.98,
                         epsilon=1e-9,
                         name='astromer_optimizer')
        # print('AdamW')
        # optimizer = AdamW(lr)

        with open(os.path.join(EXPDIR, 'config.toml'), 'w') as f:
            toml.dump(model_config, f)

        pbar  = tqdm(range(opt.num_epochs), total=opt.num_epochs)
        pbar.set_description("Epoch 0 (p={}) - rmse: -/- rsquare: -/-", refresh=True)
        pbar.set_postfix(item=0)    
        # ========= Training Loop ==================================
        es_count = 0
        min_loss = 1e9
        batch_idx = 0
        for epoch in pbar:
            pbar.set_postfix(item1=epoch)
            epoch_tr_rmse    = 0.
            epoch_tr_rsquare = 0.
            epoch_tr_loss = 0.
            epoch_vl_rmse    = 0.
            epoch_vl_rsquare = 0.
            epoch_vl_loss = 0.

            tr_counter = 0
            for numbatch, batch in enumerate(train_batches):
                pbar.set_postfix(item=numbatch)

                metrics = distributed_train_step(astromer, batch, optimizer, mirrored_strategy)
                
                epoch_tr_rmse+=metrics['rmse']
                epoch_tr_rsquare+=metrics['rsquare']
                
                tensorboard_log('rmse', metrics['rmse'], train_writer, step=batch_idx)
                tensorboard_log('rsquare', metrics['rsquare'], train_writer, step=batch_idx)
                tensorboard_log('loss', metrics['loss'], train_writer, step=batch_idx)
                batch_idx+=1
                tr_counter+=1
                
            tr_rmse    = epoch_tr_rmse/tr_counter
            tr_rsquare = epoch_tr_rsquare/tr_counter

            val_counter = 0
            for numbatch, batch in enumerate(valid_batches):
                metrics = distributed_test_step(astromer, batch, mirrored_strategy)
                epoch_vl_rmse+=metrics['rmse']
                epoch_vl_rsquare+=metrics['rsquare']
                epoch_vl_loss+=metrics['loss']

                val_counter+=1
                
            vl_rmse    = epoch_vl_rmse/val_counter
            vl_rsquare = epoch_vl_rsquare/val_counter
            vl_loss = epoch_vl_loss/numbatch 
            tensorboard_log('rmse', vl_rmse, valid_writer, step=batch_idx)
            tensorboard_log('rsquare', vl_rsquare, valid_writer, step=batch_idx)
            tensorboard_log('loss', vl_loss, valid_writer, step=batch_idx)
            
            if tf.math.greater(min_loss, vl_rmse):
                min_loss = vl_rmse
                es_count = 0
                astromer.save_weights(os.path.join(EXPDIR, 'weights'))
            else:
                es_count = es_count + 1

            if es_count == opt.patience:
                print('[INFO] Early Stopping Triggered at epoch {:03d}'.format(epoch))
                break
            
            pbar.set_description("Epoch {} (p={}) - rmse: {:.3f}/{:.3f} rsquare: {:.3f}-{:.3f}".format(epoch, 
                                                                                                es_count,
                                                                                                tr_rmse,
                                                                                                vl_rmse,
                                                                                                tr_rsquare,
                                                                                                vl_rsquare))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/records/alcock/fold_0/alcock_20', type=str,
                    help='DOWNSTREAM data path')
    parser.add_argument('--exp-name', default='finetuning', type=str,
                    help='Project name')
    parser.add_argument('--pt-model', default='-1', type=str,
                        help='Restore training by using checkpoints. This is the route to the checkpoint folder.')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--bs', default=2000, type=int,
                        help='Finetuning batch size')
    parser.add_argument('--num-epochs', default=1000000, type=int,
                        help='Finetuning batch size')
    parser.add_argument('--patience', default=20, type=int,
                        help='Finetuning batch size')
    parser.add_argument('--lr', default=0.00001, type=float,
                        help='Finetuning learning rate')

    opt = parser.parse_args()        

    ft_step(opt)
