import matplotlib.pyplot as plt
import pandas as pd
import toml
import os
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import get_metrics

from tensorboard.backend.event_processing import event_accumulator


def get_saved_metrics(path):
	with open(path, 'r') as f:
		m = toml.load(f)
		return m

def get_keys(path_logs):
    train_logs = [x for x in os.listdir(path_logs) if x.endswith('.v2')][0]
    path_train = os.path.join(path_logs, train_logs)

    ea = event_accumulator.EventAccumulator(path_train, 
                                            size_guidance={'tensors': 0})

      
    ea.Reload()

    keys = [x for x in ea.Tags()['tensors'] if 'epoch' in x]
    return keys

def get_validation_metrics(path, tag=None):
	valid_path = os.path.join(path, 'validation')
	keys = get_keys(valid_path)
	for i, k in enumerate(keys):
		df = get_metrics(valid_path, metric_name=k, full_logs=True)
		
		df = df.rename(columns={'value': k})
		if i == 0 :
			frame = df
		else:
			df = df.drop('wall_time', axis=1)
			frame = pd.merge(frame, df, how='right', on=['step'])

	if tag is not None:
		for i, t in enumerate(tag.split('_')):
			frame['col_{}'.format(i)] = [t]*frame.shape[0]
	return frame


def get_training_metrics(path, tag=None):
	train_path = os.path.join(path, 'train')

	keys = get_keys(train_path)
	for i, k in enumerate(keys):
		df = get_metrics(train_path, metric_name=k, full_logs=True)

		df = df.rename(columns={'value': k})
		if i == 0 :
			frame = df
		else:
			df = df.drop('wall_time', axis=1)
			frame = pd.merge(frame, df, how='right', on=['step'])

	if tag is not None:
		for i, t in enumerate(tag.split('_')):
			frame['col_{}'.format(i)] = [t]*frame.shape[0]
	return frame
