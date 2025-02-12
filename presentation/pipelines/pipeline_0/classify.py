import tensorflow as tf
import argparse
import pickle
import toml
import sys
import os

from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from presentation.pipelines.steps.model_design import load_pt_model, build_classifier 
from presentation.pipelines.steps.load_data import build_loader 
from presentation.pipelines.steps.metrics import evaluate_ft, evaluate_clf

def clf_step(opt, mlp_arch='avg_mlp'):
    factos = opt.data.split('/')
    ft_model = '/'.join(factos[-3:])
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    CLFDIR = os.path.join(opt.pt_model, '..', opt.exp_name, ft_model, mlp_arch)
    FTDIR  = os.path.join(opt.pt_model, '..', 'finetuning', ft_model)
    
    print('[INFO] Exp dir: ', CLFDIR)

    os.makedirs(CLFDIR, exist_ok=True)
    
    # ========= MODEL ========================================
    astromer, model_config = load_pt_model(FTDIR)

    # ========== DATA ========================================
    loaders = build_loader(opt.data, 
                           model_config, 
                           batch_size=opt.bs, 
                           clf_mode=True, 
                           normalize='zero-mean', 
                           sampling=False,
                           repeat=1,
                           return_test=True,
                          )
    
    with open(os.path.join(CLFDIR, 'config.toml'), 'w') as f:
        toml.dump(model_config, f)

    # ========== CLASIFIER ==================================== 
    model = build_classifier(astromer, 
                            model_config, 
                            opt.train_astromer, 
                            loaders['n_classes'],
                            arch=mlp_arch)
    
    model.compile(optimizer=Adam(opt.lr, 
                  name='classifier_optimizer'),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    cbks = [TensorBoard(log_dir=os.path.join(CLFDIR, 'tensorboard')),
            EarlyStopping(monitor='val_loss', patience=40),
            ModelCheckpoint(filepath=os.path.join(CLFDIR, 'weights'),
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1)]
#     print(model.get_layer('gamma_weight').variables)
    model.fit(loaders['train'], 
              epochs=1000000, 
              batch_size=opt.bs,
              validation_data=loaders['validation'],
              callbacks=cbks)
    
    metrics, y_true, y_pred = evaluate_clf(model, 
                                           loaders['test'], 
                                           model_config, 
                                           prefix='test_')

    with open(os.path.join(CLFDIR, 'test_metrics.toml'), "w") as toml_file:
        toml.dump(metrics, toml_file)

    with open(os.path.join(CLFDIR, 'predictions.pkl'), 'wb') as handle:
        pickle.dump({'true':y_true, 'pred':y_pred}, handle)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/records/alcock/fold_0/alcock_20', type=str,
                    help='DOWNSTREAM data path')
    parser.add_argument('--exp-name', default='classification', type=str,
                    help='Project name')
    parser.add_argument('--pt-model', default='-1', type=str,
                        help='Restore training by using checkpoints. This is the route to the checkpoint folder.')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--bs', default=256, type=int,
                        help='Finetuning batch size')
    parser.add_argument('--lr', default=0.00001, type=float,
                        help='Finetuning learning rate')
    parser.add_argument('--train-astromer', action='store_true', help='Train Astromer when classifying')


    opt = parser.parse_args()        
    
    
    for clf_arch in ['skip_avg_mlp']:
        clf_step(opt, clf_arch)