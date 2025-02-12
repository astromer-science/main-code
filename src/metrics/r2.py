import tensorflow as tf

@tf.function
def custom_r2(y_true, y_pred, mask):
    SS_res = tf.math.square(y_true - y_pred)
    SS_res =  tf.reduce_sum(SS_res* mask)

    valid_true = y_true*mask
    valid_mean = tf.math.divide_no_nan(tf.reduce_sum(valid_true, axis=1),
                                       tf.reduce_sum(mask, axis=1))
    
    SS_tot = tf.math.square(y_true - tf.expand_dims(valid_mean, 1))
    SS_tot = tf.reduce_sum(SS_tot*mask)

    return 1.-tf.math.divide_no_nan(SS_res, SS_tot)
