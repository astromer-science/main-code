import tensorflow as tf


@tf.function
def custom_acc(y_true, y_pred):
    num_classes = tf.shape(y_pred)[-1]
    y_pred = tf.squeeze(y_pred)
    y_pred = tf.argmax(y_pred, 1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.squeeze(y_true, -1)
    acc = tf.math.equal(y_true, y_pred)
    acc = tf.cast(acc, tf.float32)
    return tf.reduce_mean(acc)



@tf.function
def accuracy_clf(y_true, y_pred):
    num_classes = tf.shape(y_pred)[-1]
    y_pred = tf.argmax(y_pred, 1)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.argmax(y_true, 1)
    y_true = tf.cast(y_true, tf.float32)

    acc = tf.math.equal(y_true, y_pred)
    acc = tf.cast(acc, tf.float32)
    return tf.reduce_mean(acc)