import tensorflow as tf

@tf.function
def custom_bce(y_true, y_pred):
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_one  = tf.one_hot(y_true, num_classes)
    y_one  = tf.cast(y_one, tf.float32)

    y_one  = tf.squeeze(y_one, axis=1)
    y_pred = tf.squeeze(y_pred, axis=1)
    losses = tf.nn.softmax_cross_entropy_with_logits(y_one, y_pred)

    return tf.reduce_mean(losses)
