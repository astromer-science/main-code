import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense


class RegLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reg_layer = Dense(1, name='RegLayer')
		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		x = self.bn_0(inputs)
		x = self.reg_layer(x)
		return x

class SauceLayer(tf.keras.layers.Layer):
	def init(self, shape,**kwargs):
	    super(SauceLayer, self).init(**kwargs)
	    self.supports_masking = True
	    self.shape = shape

	def build(self, input_shape):
	    self.scale = tf.Variable([1/self.shape for _ in range(self.shape)], trainable=True)

	def call(self, inputs):
	    # Softmax normalized
	    scale = tf.nn.softmax(self.scale)
	    return tf.tensordot(scale, inputs, axes=1)
