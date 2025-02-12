import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense, TimeDistributed


class TransformLayer_GAP(Layer):
	def __init__(self, **kwargs):
		super(TransformLayer_GAP, self).__init__(**kwargs)
		self.gap_layer = Dense(1, activation='relu', name='Gap')
		self.reg_layer = Dense(1, name='Reconstruction')

		self.long_reg_layer = Dense(1, name='LongRec')

		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs, training):

		cls_token = tf.slice(inputs, [0, 0, 0], [-1, 1, -1], name='cls_token')
		rec_token = tf.slice(inputs, [0, 1, 0], [-1, -1, -1], name='rec_token')

		x_gap = self.gap_layer(cls_token)
		rec_token = self.bn_0(rec_token, training=training)
		x_rec = self.reg_layer(rec_token)
		x_gap = tf.squeeze(x_gap)

		dt = tf.reshape(x_gap, [-1, 1, 1])
		dt = tf.tile(dt, [1, tf.shape(rec_token)[1], 1])
		rec_dt = tf.concat([dt, rec_token], axis=-1)
		x_gap_rec = self.long_reg_layer(rec_dt)

		return x_gap, x_rec, x_gap_rec

class TransformLayer(Layer):
	def __init__(self, **kwargs):
		super(TransformLayer, self).__init__(**kwargs)
		self.clf_layer = Dense(2, name='Classification')
		self.reg_layer = Dense(1, name='Reconstruction')
		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs, training):

		cls_token = tf.slice(inputs, [0, 0, 0], [-1, 1, -1], name='cls_token')
		rec_token = tf.slice(inputs, [0, 1, 0], [-1, -1, -1], name='rec_token')

		x_prob = self.clf_layer(cls_token)
		rec_token = self.bn_0(rec_token, training=training)
		x_rec = self.reg_layer(rec_token)

		return {'nsp_label': x_prob, 'reconstruction':x_rec}

class RegLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reg_layer = Dense(1, name='reconstruction')
		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs, training=False):
		x = self.bn_0(inputs, training=training)
		x = self.reg_layer(x)
		return x

class UpRegLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reg_layer = TimeDistributed(Dense(1, name='reconstruction'))
		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs, training):
		x = tf.expand_dims(inputs[0], axis=1)
		t = inputs[1]
		x = tf.tile(x, [1, tf.shape(t)[1], 1])
		x = tf.concat([t, x], axis=-1)
		x = self.bn_0(x, training=training)
		x = self.reg_layer(x)
		return x