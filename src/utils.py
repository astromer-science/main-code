import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import toml
import json
import os

from tensorboard.backend.event_processing import event_accumulator
from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2
from datetime import datetime


def plot_cm(cm, ax, title='CM', fontsize=15, cbar=False, yticklabels=True, class_names=None):
    '''
    Plot Confusion Matrix
    '''
    labels = np.zeros_like(cm, dtype=np.object)
    mask = np.ones_like(cm, dtype=np.bool)
    for (row, col), value in np.ndenumerate(cm):
        if value != 0.0:
            mask[row][col] = False
        if value < 0.01:
            labels[row][col] = '< 1%'
        else:
            labels[row][col] = '{:2.1f}\\%'.format(value*100)

    ax = sns.heatmap(cm, annot = labels, fmt = '',
                     annot_kws={"size": fontsize},
                     cbar=cbar,
                     ax=ax,
                     linecolor='white',
                     linewidths=1,
                     vmin=0, vmax=1,
                     cmap='Blues',
                     mask=mask,
                     yticklabels=yticklabels)

    try:
        if yticklabels and class_names is not None:
            ax.set_yticklabels(class_names, rotation=0, fontsize=fontsize+1)
            ax.set_xticklabels(class_names, rotation=0, fontsize=fontsize+1)
    except:
        pass
    ax.set_title(title, fontsize=fontsize+5)

    ax.axhline(y=0, color='k',linewidth=4)
    ax.axhline(y=cm.shape[1], color='k',linewidth=4)
    ax.axvline(x=0, color='k',linewidth=4)
    ax.axvline(x=cm.shape[0], color='k',linewidth=4)

    return ax

def get_folder_name(path, prefix=''):
    """
    Look at the current path and change the name of the experiment
    if it is repeated

    Args:
        path (string): folder path
        prefix (string): prefix to add

    Returns:
        string: unique path to save the experiment
"""

    if prefix == '':
        prefix = path.split('/')[-1]
        path = '/'.join(path.split('/')[:-1])

    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    if prefix not in folders:
        path = os.path.join(path, prefix)
    elif not os.path.isdir(os.path.join(path, '{}_0'.format(prefix))):
        path = os.path.join(path, '{}_0'.format(prefix))
    else:
        n = sorted([int(f.split('_')[-1]) for f in folders if '_' in f[-2:]])[-1]
        path = os.path.join(path, '{}_{}'.format(prefix, n+1))

    return path

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

def get_metrics(path_logs, metric_name='epoch_loss', full_logs=True, show_keys=False, nlog=-1):
    train_logs = [x for x in os.listdir(path_logs) if x.endswith('.v2')][nlog]

    path_train = os.path.join(path_logs, train_logs)

    if full_logs:
        ea = event_accumulator.EventAccumulator(path_train, 
                                                size_guidance={'tensors': 0})
    else:
        ea = event_accumulator.EventAccumulator(path_train)
      
    ea.Reload()

    if show_keys:
        print(ea.Tags())

    try:
        metrics = pd.DataFrame([(w,s,tf.make_ndarray(t)) for w,s,t in ea.Tensors(metric_name)],
                    columns=['wall_time', 'step', 'value'])
    except:
        metrics = pd.DataFrame([(x.wall_time, x.step, tf.make_ndarray(x.tensor_proto)) for x in ea.Tensors(metric_name)],
                                columns=['wall_time', 'step', 'value'])
        
    return metrics

def tensorboard_logs(folder):
    config_file = os.path.join(folder, 'config.toml')
    with open(config_file, 'r') as file:
        config = toml.load(file)  
      
    path_logs = os.path.join(folder, 'tensorboard', 'validation')
    train_logs = [x for x in os.listdir(path_logs) if x.endswith('.v2')][-1]
    ea = event_accumulator.EventAccumulator(os.path.join(path_logs, train_logs))
    ea.Reload()

    if 'paper' in folder:
        metric_names = ea.Tags()['tensors']

    else:
        metric_names = ea.Tags()['tensors']

    output = []
    for sset in ['train', 'validation']:
        try:
            sset_df = []
            for metric in metric_names:
                df = get_metrics(os.path.join(folder, 'tensorboard', sset), metric_name=metric, show_keys=False)
                df = df.rename(columns={'value': metric.split('_')[-1]})
                sset_df.append(df.iloc[:, -1])
                
            curr = pd.concat(sset_df, axis=1)
            general = df.iloc[:, :-1]
            curr = pd.concat([general, curr], axis=1)
            curr['exp_name']    = [config['exp_name']]*curr.shape[0] 
            curr['data']        = [config['data']]*curr.shape[0] 
            curr['probed']      = [config['probed']]*curr.shape[0] 
            curr['rs']          = [config['rs']]*curr.shape[0] 
            curr['arch']        = [config['arch']]*curr.shape[0] 
            curr['m_alpha']     = [config['m_alpha']]*curr.shape[0] 
            curr['mask_format'] = [config['mask_format']]*curr.shape[0] 
            curr['temperature'] = [config['temperature']]*curr.shape[0] 
            curr['lr']          = [config['lr']]*curr.shape[0] 
            curr['scheduler']   = [config['scheduler']]*curr.shape[0] 
            curr['leak']        = [config['use_leak']]*curr.shape[0] 
            
            if 'paper' in folder:
                curr['mse'] = curr['mse']
                curr['loss'] = curr['mse']
                curr['square'] = np.zeros_like(curr['mse'])
                curr = curr.rename(columns={'mse':'rmse'})
                
            output.append(curr)
        except:
            output.append([])

        
    return output

def dict_to_json(varsdic, conf_file):
    now = datetime.now()
    varsdic['exp_date'] = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(conf_file, 'w') as json_file:
        json.dump(varsdic, json_file, indent=4)
        
    
    
    
    
    
