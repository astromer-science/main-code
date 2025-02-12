import toml
import pandas as pd
import numpy as np
import os


def classification_metrics(root, sset=['alcock', 'atlas'], spc_list=[20, 100], n_folds=3, clf_arch='avg_mlp'):
    config_file = os.path.join(root, 'pretraining', 'config.toml')
    with open(config_file, 'r') as file:
        config = toml.load(file)  
        
    rows = []
    for sset in sset:
        for spc in spc_list:
            fold_metrics = []
            for fold_n in range(n_folds):
                tfile = os.path.join(root, 
                                     'classification', 
                                     sset, 
                                     'fold_{}'.format(fold_n), 
                                     '{}_{}'.format(sset, spc), 
                                     clf_arch, 
                                     'test_metrics.toml')
                try:
                    with open(tfile, 'r') as handle:
                        test_metrics = toml.load(handle)
                        f1_test = test_metrics['test_f1']
                        fold_metrics.append(float(f1_test))
                except:
                    fold_metrics.append(-1.)

            valid_values = [x for x in fold_metrics if x != -1]

            rows.append({'exp_name': config['exp_name'],
                         'probed': config['probed'],
                         'rs': config['rs'],
                         'arch': config['arch'],
                         'm_alpha': config['m_alpha'],
                         'mask_format': config['mask_format'],
                         'temperature': config['temperature'],
                         'data':sset, 
                         'spc': spc, 
                         'mean': np.mean(valid_values), 
                         'std': np.std(valid_values)})

    return pd.DataFrame(rows)