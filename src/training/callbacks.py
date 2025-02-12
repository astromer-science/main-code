import tensorflow as tf
import os

from .utils import average_logs

class SaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, frequency, project_path, **kwargs):
        super(SaveCheckpoint, self).__init__(**kwargs)
        self.frequency = frequency
        self.project_path = project_path
        self.best_model = self.model
        self.best_loss = 1e9

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best_loss:
            self.best_loss = logs['val_loss']
            self.best_model = self.model
        
        if self.frequency is None:
            self.best_model.save_weights(os.path.join(self.project_path, 'weights', 'weights'))
        else:
            if epoch % self.frequency == 0:
                self.model.save_weights(os.path.join(self.project_path, 'ckpts', 'epoch_{}'.format(epoch), 'weights', 'weights'))


            
class TestModel(tf.keras.callbacks.Callback):
    def __init__(self, test_batches, project_path, test_step_fn, params, **kwargs):
        super(TestModel, self).__init__(**kwargs)
        self.project_path = project_path
        self.best_model = self.model
        self.best_loss = 1e9
        self.test_batches = test_batches
        self.writer = tf.summary.create_file_writer(os.path.join(project_path, 'test'), name='test_writer')
        self.last_epoch = 0
        self.test_step_fn = test_step_fn
        self.params = params

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best_loss:
            self.best_loss = logs['val_loss']
            self.best_model = self.model        
        self.last_epoch = epoch

    def on_train_end(self, logs=None):
        print('[INFO] Testing step')
        self.best_model.save_weights(os.path.join(self.project_path, '..', 'weights', 'weights'))

        if not 'rmse_factor' in self.params.keys():
            self.params['rmse_factor'] = 1.

        test_logs = []
        for x, y in self.test_batches:
            logs = self.test_step_fn(self.best_model, x, y, rmse_factor=self.params['rmse_factor'])
            test_logs.append(logs)
        test_metrics = average_logs(test_logs)
        with self.writer.as_default():
            for key, value in test_metrics.items():
                tf.summary.scalar('epoch_{}'.format(key), value, step=0)
                tf.summary.scalar('epoch_{}'.format(key), value, step=self.last_epoch)