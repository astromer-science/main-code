import tensorflow as tf


# ==== PIPELINE ====
# Given a set of Hyperparameter we want to 
# Start pretraining a model
# Capture validation R2 and RMSE
# At a given number of batches, evaluate on classification 
# Capture F1 SCORE

import argparse
import wandb
import os
from tqdm import tqdm


from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from functools import partial

from presentation.pipelines.steps.model_design import build_model, load_pt_model, build_classifier
from presentation.pipelines.steps.load_data import build_loader
from presentation.pipelines.steps.metrics import evaluate_clf

from src.training.utils import train_step, test_step
from src.data.loaders import get_loader
from src.losses.rmse import custom_rmse
from src.metrics import custom_r2


def replace_param(params, replace):
    for key in replace.keys():
        print('[INFO] REPLACING {}'.format(key))
        params[key] = replace[key]

    return params

def train(config=None, params=None):
    with wandb.init(config=config):
        curr_config = wandb.config
        params = replace_param(params, curr_config)
       
        val_loader = get_loader(os.path.join(params['data'], 'validation'),
                                batch_size=params['bs'],
                                window_size=params['window_size'],
                                probed_frac=params['probed'],
                                random_frac=params['rs'],
                                same_frac=params['same'],
                                sampling=True,
                                shuffle=False,
                                normalize='zero-mean',
                                repeat=0,
                                cache=True,
                                aversion='base')
        
        train_loader = get_loader(os.path.join(params['data'], 'train'),
                            batch_size=params['bs'],
                            window_size=params['window_size'],
                            probed_frac=params['probed'],
                            random_frac=params['rs'],
                            same_frac=params['same'],
                            sampling=False,
                            shuffle=True,
                            normalize='zero-mean',
                            repeat=0,
                            aversion='base')
        clf_loaders = build_loader(params['downstream_data'], 
                                   params, 
                                   batch_size=params['bs'], 
                                   clf_mode=True, 
                                   normalize='zero-mean', 
                                   sampling=True,
                                   repeat=1,
                                   return_test=True)

        if params['debug']:
            print('DEBUG')
            train_loader = train_loader.take(1)
            val_loader = val_loader.take(1)
            for k in  ['train', 'validation', 'test']:
                try:
                    clf_loaders[k] = clf_loaders[k].take(1)
                except:
                    continue
                    
            params['num_epochs'] = 1
            
        astromer = build_model(params)


        optimizer = Adam(params['lr'], 
                         beta_1=0.9,
                         beta_2=0.98,
                         epsilon=1e-9,
                         name='astromer_optimizer') 
        
        pbar  = tqdm(range(params['num_epochs']), total=params['num_epochs'])
        # pbar.set_description("Epoch 0 (p={}) - rmse: -/- rsquare: -/-", refresh=True)
        # pbar.set_postfix(item=0)    
        for epoch in pbar:
            pbar.set_postfix(item1=epoch)
            for numbatch, batch in enumerate(train_loader):
                pbar.set_postfix(item=numbatch)
                metrics = train_step(astromer, batch, optimizer)
                wandb.log({"batch_loss": metrics['loss'],
                           "batch_rmse": metrics['rmse'],
                           "batch_rsquare": metrics['rsquare']})
                
            metrics = val_step(astromer, val_loader)
            wandb.log(metrics)
            val_loss, val_acc = clf_step(astromer, params, loaders=clf_loaders)       
            wandb.log({"val_acc": val_acc, 'val_cce': val_loss})
    

    
def val_step(astromer, loader):
    val_loss, val_rmse, val_rsquare = 0., 0., 0.
    for numbatch, batch in enumerate(loader):
        metrics = test_step(astromer, batch)
        val_loss+=metrics['loss']
        val_rmse+=metrics['rmse']
        val_rsquare+=metrics['rsquare']
    val_loss = val_loss/numbatch
    val_rmse = val_rsquare/numbatch
    val_rsquare = val_rmse/numbatch
    return {"val_loss": val_loss,
             "val_rmse": val_rmse,
             "val_rsquare": val_rsquare}  


def clf_step(astromer, params, loaders):    
    # ========== CLASIFIER ==================================== 
    model = build_classifier(astromer, 
                             params, 
                             False, 
                             loaders['n_classes'],
                             arch='avg_mlp',
                             verbose=0)
    
    model.compile(optimizer=Adam(opt.lr, 
                  name='classifier_optimizer'),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.fit(loaders['train'], 
              epochs=50, 
              batch_size=512,
              validation_data=loaders['validation'],
              verbose=0)
    
    val_loss, val_acc = model.evaluate(loaders['test'])
    return val_loss, val_acc


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
    parser.add_argument('--num_epochs', default=500, type=int,
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

    parser.add_argument('--sweep-id', default='', type=str,
                        help='SWEEP ID')

    opt = parser.parse_args()        

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    preloaded = partial(train, params=opt.__dict__)


    wandb.login()
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters':{
            'm_alpha': {'values': [-1000000000., -1000., -100., -10., -1., 0., 1.]},
            'temperature': {'distribution': 'uniform', 'min': 1, 'max':5},
            'no_msk_token': {'values': [True, False]},
            'downstream_data': {'values': ['./data/shared/records/alcock/fold_0/alcock_20',
                                           './data/shared/records/atlas/fold_0/atlas_20']}
        },
        'early_terminate': {
            "type": "hyperband",
            "eta": 2,
            "min_iter":2,
            "max_iter":opt.num_epochs
        }
    }

    sweep_id = wandb.sweep(sweep_config, 
                project="pipeline_0")

    wandb.agent(sweep_id, preloaded, count=1)
