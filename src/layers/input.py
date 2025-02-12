import tensorflow as tf

from tensorflow.keras.layers import Layer, Softmax

class GammaWeight(Layer):
    """ Weighted trainable average between attention outputs and input embeddings """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.softmax_act = Softmax(name='softmax')
        
    def build(self, input_shape):
        initial_value = tf.ones([input_shape[0]]) * 1/input_shape[0]
        self.gamma = tf.Variable(initial_value=initial_value,
                                 trainable=True,
                                 name='gamma')
        
    def call(self, inputs, training):
        gamma = self.softmax_act(tf.expand_dims(self.gamma, 0))
        gamma = tf.reshape(gamma, [tf.shape(inputs)[0], 1, 1])
        x = tf.multiply(inputs, gamma)
        x = tf.reduce_mean(x, axis=0)
        return x
    
class AddMSKToken(Layer):
    """ Create MSK token """
    def __init__(self, 
                 trainable=True,
                 window_size=200,
                 on=['input'],
                 **kwargs):

        super().__init__(**kwargs)

        self.trainable = trainable
        self.on = on
        self.window_size = window_size

    def build(self, input_shape):
        self.msk_token = tf.Variable(
                        initial_value=tf.constant([[0.]]),
                        dtype=tf.float32,
                        trainable=self.trainable,)

    def call(self, inputs):
        msk_token = tf.tile(self.msk_token, [self.window_size, 1])
        for key in self.on:
            partial = tf.multiply(inputs[key], 1.-inputs['mask_in'])
            partial_mask = tf.multiply(inputs['mask_in'], msk_token)
            partial = tf.add(partial, partial_mask)
            inputs[key] = partial 
        return inputs


