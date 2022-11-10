import tensorflow as tf
import pandas as pd
import os

from tensorboard.backend.event_processing import event_accumulator
from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2


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

def standardize(tensor, axis=0, return_mean=False):
    """
    Standardize a tensor subtracting the mean

    Args:
        tensor (1-dim tensorflow tensor): values
        axis (int): axis on which we calculate the mean
        return_mean (bool): output the mean of the tensor
                            turning on the original scale
    Returns:
        tensor (1-dim tensorflow tensor): standardize tensor
    """
    mean_value = tf.reduce_mean(tensor, axis, name='mean_value')
    z = tensor - tf.expand_dims(mean_value, axis)

    if return_mean:
        return z, mean_value
    else:
        return z
    
def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)
        
def get_metrics(path_logs, metric_name='epoch_loss', full_logs=True):
    train_logs = [x for x in os.listdir(path_logs) if x.endswith('.v2')][0]
    path_train = os.path.join(path_logs, train_logs)
    
    if full_logs:
        ea = event_accumulator.EventAccumulator(path_train, size_guidance={'tensors': 0})
    else:
        ea = event_accumulator.EventAccumulator(path_train)

    ea.Reload()   
    
    metrics = pd.DataFrame([(w,s,tf.make_ndarray(t))for w,s,t in ea.Tensors(metric_name)],
                columns=['wall_time', 'step', 'value'])
    return metrics