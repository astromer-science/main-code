import tensorflow as tf 

class MeanSquaredError(Loss):

  def call(self, y_true, y_pred):
      return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)