import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, LayerNormalization

from src.losses import custom_rmse
from src.metrics import custom_r2
from src.layers.positional import positional_encoding


def scaled_dot_product_attention(q, k, v, mask, m_alpha, mask_format='QK'):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    qkvalues = scaled_attention_logits
    
    if mask_format == 'Q':
        steps = tf.shape(scaled_attention_logits)[2]
        mask_rshp = tf.tile(mask, [1, 1, steps])
        mask_rshp = tf.transpose(mask_rshp, [0, 2, 1])
        mask_rshp = tf.minimum(1., mask_rshp)
        mask_rshp = tf.expand_dims(mask_rshp, 1)
        scaled_attention_logits += (mask_rshp * m_alpha)
    
    if mask_format == 'QK':
        steps = tf.shape(scaled_attention_logits)[2]
        mask_rshp = tf.tile(mask, [1, 1, steps])
        mask_rshp += tf.transpose(mask_rshp, [0, 2, 1])
        mask_rshp = tf.minimum(1., mask_rshp)
        mask_rshp = tf.expand_dims(mask_rshp, 1)
        scaled_attention_logits += (mask_rshp * m_alpha)
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')
    output = tf.matmul(attention_weights, v, name='Z')

    return output, attention_weights, qkvalues


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, m_alpha, mask_format):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.m_alpha = m_alpha
        self.mask_format = mask_format

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads 
        self.wq = Dense(d_model, name='WQ')
        self.wk = Dense(d_model, name='WK')
        self.wv = Dense(d_model, name='WV')
        self.dense = Dense(d_model, name='MixerDense')

    def split_heads(self, x, batch_size, name='qkv'):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

    def call(self, x, mask):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size, name='Q')
        k = self.split_heads(k, batch_size, name='K')
        v = self.split_heads(v, batch_size, name='V')

        scaled_attention, attention_weights, qkvalues = scaled_dot_product_attention(
            q, k, v, mask=mask, m_alpha=self.m_alpha, mask_format=self.mask_format)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights, qkvalues, (q, k, v)


class RegLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reg_layer = Dense(1, name='RegLayer')
        self.bn_0 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.bn_0(inputs)
        x = self.reg_layer(x)
        return x
    

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        Dense(dff, activation='tanh'),
        Dense(d_model)
    ])


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, use_leak=False, m_alpha=1., mask_format='QK', **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads, m_alpha, mask_format)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.use_leak = use_leak
        if use_leak:
            self.reshape_leak_1 = Dense(d_model)
            self.reshape_leak_2 = Dense(d_model)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask, return_weights=False):
        attn_output, w, qkvalues, (q,k,v) = self.mha(x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        if self.use_leak:
            out1 = self.layernorm1(self.reshape_leak_1(x) + attn_output)
        else:
            out1 = self.layernorm1(attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        if self.use_leak:
            out2 = self.layernorm2(self.reshape_leak_2(out1) + ffn_output)
        else:
            out2 = self.layernorm2(ffn_output)
        
        if return_weights:
            return out2, w, qkvalues
        return out2


class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 base=10000, rate=0.1, use_leak=False, m_alpha=1., mask_format='QK', return_weights=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.base = base
        self.inp_transform = Dense(d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, use_leak, m_alpha, mask_format)
                            for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, data, training=False, return_weights=False, z_by_layer=False):
        # adding embedding and position encoding.
        x_pe = positional_encoding(data['times'], self.d_model, mjd=True)

        x_transformed = self.inp_transform(data['input'])
        transformed_input = x_transformed + x_pe

        x = self.dropout(transformed_input, training=training)

        output_by_layer = []
        for i in range(self.num_layers):
            if return_weights:
                x, w, qkvalues = self.enc_layers[i](x, training=training, mask=data['mask_in'], 
                                                    return_weights=True)
            else:
                x = self.enc_layers[i](x, training=training, mask=data['mask_in'], return_weights=False)
            output_by_layer.append(x)
        
        if return_weights:
            return x, w, qkvalues
        
        if z_by_layer:
            return output_by_layer
        return x


def build_input(length):
    serie = Input(shape=(length, 1), batch_size=None, name='input')
    times = Input(shape=(length, 1), batch_size=None, name='times')
    mask  = Input(shape=(length, 1), batch_size=None, name='mask')

    return {'input': serie,
            'mask_in': mask,
            'times': times}


def get_ASTROMER(num_layers=2,
                 d_model=200,
                 num_heads=2,
                 dff=256,
                 base=10000,
                 rate=0.1,
                 use_leak=False,
                 maxlen=100,
                 batch_size=None,
                 m_alpha=1,
                 mask_format='QK',
                 return_weights=False,
                 loss_format='mse',
                 correct_loss=False,
                 temperature=0.):

    placeholder = build_input(maxlen)

    encoder = Encoder(num_layers,
                      d_model,
                      num_heads,
                      dff,
                      base=base,
                      rate=rate,
                      use_leak=False,
                      name='encoder',
                      m_alpha=m_alpha,
                      mask_format=mask_format,
                      return_weights=return_weights)
    
    if return_weights:
        x, w, qkvalues = encoder(placeholder, return_weights=True)
    else:
        x = encoder(placeholder, return_weights=False)
        
    x = RegLayer(name='regression')(x)
    
    if return_weights:
        return CustomModel(inputs=placeholder, 
                           outputs={'output': x, 'w': w, 'qk_values': qkvalues}, 
                           name="ASTROMER")
    
    return CustomModel(correct_loss=correct_loss, loss_format=loss_format, inputs=placeholder, outputs=x, name="ASTROMER-0")


### KERAS MODEL 
class CustomModel(Model):
    def __init__(self, correct_loss=False, loss_format='mse', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_format = loss_format
        self.correct_loss = correct_loss
        
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            x_pred = self(x, training=True) 
            rmse = custom_rmse(y_true=y['target'],
                              y_pred=x_pred,
                              mask=y['mask_out'],
                              root=False if self.loss_format == 'mse' else True)
            r2_value = custom_r2(y['target'], x_pred, mask=y['mask_out'])

        grads = tape.gradient(rmse, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': rmse, 'r_square': r2_value, 'rmse': rmse}

    def test_step(self, data):
        x, y = data
        x_pred = self(x, training=False)
        rmse = custom_rmse(y_true=y['target'],
                          y_pred=x_pred,
                          mask=y['mask_out'],
                          root=False if self.loss_format == 'mse' else True)
        r2_value = custom_r2(y['target'], x_pred, mask=y['mask_out'])
        return {'loss': rmse, 'r_square': r2_value, 'rmse': rmse}
    
    def predict_step(self, data):
        x, y = data
        x_pred = self(x, training=False)
        
        return {'reconstruction': x_pred, 
                'magnitudes': y['target'],
                'times': x['times'],
                'mask_in': x['mask_in'],
                'probed_mask': y['mask_out']}