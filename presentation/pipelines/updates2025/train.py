import tensorflow as tf
import argparse
import pickle
import toml
import os

from presentation.pipelines.steps import model_design
from presentation.pipelines.steps import build_loader
from presentation.pipelines.referee import classifiers, baseline_clf
from presentation.pipelines.steps.metrics import evaluate_clf

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


# https://github.com/astromer-science/weights/raw/refs/heads/main/macho_a1.zip

def clf_step(opt):
    root = './presentation/pipelines/referee/output'
    CLFDIR = os.path.join(root, opt.exp_name, opt.clf_arch)
    os.makedirs(CLFDIR, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    # Load pretrained model
    
    pt_model, pt_config = model_design.load_pt_model(opt.pt_path, optimizer=None)

    # Load data 
    loaders = build_loader(opt.data, 
                           pt_config, 
                           batch_size=opt.bs, 
                           clf_mode=True, 
                           normalize='zero-mean', 
                           sampling=False,
                           repeat=1,
                           return_test=True,
                          )


    pt_config['embedding_dim'] = pt_config['num_heads']*pt_config['head_dim']
    
    # Build Classifiers
    pt_config['num_cls'] = loaders['n_classes']

    if opt.clf_arch == 'max':
        classifier = classifiers.max_clf(pt_model, pt_config)

    if opt.clf_arch == 'avg':
        classifier = classifiers.avg_clf(pt_model, pt_config)

    if opt.clf_arch == 'skip':
        classifier = classifiers.skip_avg_clf(pt_model, pt_config)

    if opt.clf_arch == 'att_avg':
        classifier = classifiers.att_avg(pt_model, pt_config)

    if opt.clf_arch == 'att_cls':
        classifier = classifiers.att_cls(pt_model, pt_config)

    if opt.clf_arch == 'base_avgpool':
        print('BASE AVG POOL')
        classifier = baseline_clf.build_supervised_pooling_classifier(pt_config)    

    if opt.clf_arch == 'base_gru':
        print('BASE GRU')
        classifier = baseline_clf.build_supervised_rnn_classifier(pt_config)
        
    if opt.clf_arch == 'raw_gru':
            print('RAW GRU (No Embeddings)')
            classifier = baseline_clf.build_raw_rnn_classifier(pt_config)

    # Compile and train
    classifier.compile(optimizer=Adam(opt.lr, 
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

    import numpy as np
    for x, y in loaders['train']: 
        print(np.unique(np.argmax(y, 1), return_counts=True))

    classifier.fit(loaders['train'], 
                epochs=opt.num_epochs, 
                batch_size=opt.bs,
                validation_data=loaders['validation'],
            callbacks=cbks)

    metrics, y_true, y_pred = evaluate_clf(classifier, 
                                           loaders['test'], 
                                           pt_config, 
                                           prefix='test_')

    with open(os.path.join(CLFDIR, 'test_metrics.toml'), "w") as toml_file:
        toml.dump(metrics, toml_file)

    with open(os.path.join(CLFDIR, 'predictions.pkl'), 'wb') as handle:
        pickle.dump({'true':y_true, 'pred':y_pred}, handle)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', 
                        default='./data/records/alcock/fold_0/', 
                        type=str,
                        help='DOWNSTREAM data path')
    parser.add_argument('--exp-name', 
                        default='classification', 
                        type=str,
                        help='Project name')
    parser.add_argument('--pt-path', 
            default='./presentation/pipelines/referee/weights/astromer_2/macho-clean', 
            type=str,
            help='Pretrained weights')
    parser.add_argument('--clf-arch', 
            default='skip', 
            type=str,
            help='Pretrained weights')
    
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--num-epochs', default=10000000, type=int,
                        help='Finetuning batch size')
    parser.add_argument('--bs', default=256, type=int,
                        help='Finetuning batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Finetuning learning rate')

    opt = parser.parse_args()        
    
    clf_step(opt)