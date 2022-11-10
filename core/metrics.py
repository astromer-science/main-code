import tensorflow as tf


@tf.function
def custom_acc(y_true, y_pred):
    if len(tf.shape(y_pred)) > 2:
        y_pred  = tf.nn.softmax(y_pred)[:,-1,:]
    else:
        y_pred  = tf.nn.softmax(y_pred)

    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.argmax(y_pred, 1, output_type=tf.int32)
    y_pred = tf.expand_dims(y_pred, 1)

    correct = tf.math.equal(y_true, y_pred)
    correct = tf.cast(correct, tf.float32)

    return tf.reduce_mean(correct)
