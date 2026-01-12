import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from src.losses import custom_rmse
from src.metrics import custom_r2
from src.layers import Encoder, RegLayer
from src.layers.input import AddMSKToken


def build_input(window_size, batch_size=None):
    magnitudes = Input(shape=(window_size, 1),
                       batch_size=batch_size,
                       name='input')
    times = Input(shape=(window_size, 1),
                  batch_size=batch_size,
                  name='times')
    att_mask = Input(shape=(window_size, 1),
                     batch_size=batch_size,
                     name='mask_in') 

    return {'input': magnitudes,
            'times': times,
            'mask_in': att_mask}


def get_ASTROMER(num_layers=2,
                 num_heads=2,
                 head_dim=64,
                 mixer_size=256,
                 dropout=0.1,
                 pe_base=1000,
                 pe_dim=128,
                 pe_c=1,
                 window_size=100,
                 batch_size=None,
                 m_alpha=-0.5,
                 mask_format='Q',
                 use_leak=False,
                 loss_format='rmse',
                 correct_loss=False,
                 trainable_mask=True,
                 temperature=0.):

    placeholder = build_input(window_size, batch_size)

    if trainable_mask:
        placeholder = AddMSKToken(trainable=True, 
                                  window_size=window_size, 
                                  on=['input'], 
                                  name='msk_token')(placeholder)

    encoder = Encoder(window_size=window_size,
                      num_layers=num_layers,
                      num_heads=num_heads,
                      head_dim=head_dim,
                      mixer_size=mixer_size,
                      dropout=dropout,
                      pe_base=pe_base,
                      pe_dim=pe_dim,
                      pe_c=pe_c,
                      m_alpha=m_alpha,
                      mask_format=mask_format,
                      use_leak=use_leak,
                      temperature=temperature,
                      name='encoder')

    x = encoder(placeholder)
    x = RegLayer(name='regression')(x)

    return CustomModel(correct_loss=correct_loss, 
                       loss_format=loss_format, 
                       inputs=placeholder, 
                       outputs=x, 
                       name="ASTROMER-1")


class CustomModel(Model):
    def __init__(self, correct_loss=False, loss_format='rmse', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_format = loss_format
        self.correct_loss = correct_loss
        
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            use_root = (self.loss_format == 'rmse')
            
            rmse = custom_rmse(y_true=y['target'],
                               y_pred=y_pred,
                               mask=y['mask_out'],
                               weights=y['w_error'] if self.correct_loss else None,
                               root=use_root)
            
            r2_value = custom_r2(y_true=y['target'], 
                                 y_pred=y_pred, 
                                 mask=y['mask_out'])
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(rmse, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {'loss': rmse, 'r_square': r2_value, 'rmse': rmse}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        
        use_root = (self.loss_format == 'rmse')
        
        rmse = custom_rmse(y_true=y['target'],
                           y_pred=y_pred,
                           mask=y['mask_out'],
                           weights=y['w_error'] if self.correct_loss else None,
                           root=use_root)
        
        r2_value = custom_r2(y_true=y['target'], 
                             y_pred=y_pred, 
                             mask=y['mask_out'])
        
        return {'loss': rmse, 'r_square': r2_value, 'rmse': rmse}
    
    def predict_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        
        return {'reconstruction': y_pred, 
                'magnitudes': y['target'],
                'times': x['times'],
                'probed_mask': y['mask_out'],
                'mask_in': x['mask_in']}