import tensorflow as tf

from tensorflow.keras.layers import LSTMCell, LayerNormalization, Layer

def build_zero_init_state(x, state_size):
	s0 = [tf.zeros([tf.shape(x)[0], state_size]),
		  tf.zeros([tf.shape(x)[0], state_size])]
	s1 = [tf.zeros([tf.shape(x)[0], state_size]),
		  tf.zeros([tf.shape(x)[0], state_size])]
	return [s0, s1]

class NormedLSTMCell(Layer):

	def __init__(self, units, **kwargs):
		self.units = units
		self.state_size = ((self.units, self.units), (self.units, self.units))

		super(NormedLSTMCell, self).__init__(**kwargs)

		self.cell_0 = tf.keras.layers.LSTMCell(self.units)
		self.cell_1 = tf.keras.layers.LSTMCell(self.units)
		self.bn = LayerNormalization(name='bn_step')

	def call(self, inputs, states, training=False):
		s0, s1 = states[0], states[1]
		output, s0 = self.cell_0(inputs, states=s0, training=training)
		output = self.bn(output, training=training)
		output, s1 = self.cell_1(output, states=s1, training=training)
		return output, [s0, s1]

	def get_config(self):
		config = super(NormedLSTMCell, self).get_config().copy()
		config.update({"units": self.units})
		return config