import tensorflow as tf

@tf.function
def custom_rmse(y_true, y_pred, mask=None, weights=None, root=True):
    inp_shp = tf.shape(y_true)
    
    residuals = tf.square(y_true - y_pred)
    if not weights is None:
        residuals = tf.multiply(residuals, weights)
    if mask is not None:
        residuals = tf.multiply(residuals, mask)
        residuals  = tf.reduce_sum(residuals, 1)
        mse_mean = tf.math.divide_no_nan(residuals, tf.reduce_sum(mask, 1))
    else:
        mse_mean = tf.reduce_mean(residuals, 1)
        
    mse_mean = tf.reduce_mean(mse_mean)
    
    if root:
        return tf.math.sqrt(mse_mean)
    else:
        return mse_mean
    
def pearson_loss(y_true, y_pred, mask):
    def fn(a, b, mask):
        a = tf.boolean_mask(a, mask)
        b = tf.boolean_mask(b, mask)
        
        a_mean = tf.reduce_mean(a)
        b_mean = tf.reduce_mean(b)
        
        num = tf.math.multiply_no_nan(a - a_mean, b - b_mean)
        num = tf.reduce_sum(num)
        
        fac_0 = tf.reduce_sum(tf.pow(a - a_mean, 2))
        fac_1 = tf.reduce_sum(tf.pow(b - b_mean, 2))
        den = tf.math.multiply_no_nan(fac_0, fac_1)
        den = tf.math.sqrt(den)
        
        corr = tf.math.divide_no_nan(num, den)
        return corr
    vals = tf.map_fn(lambda x: fn(x[0], x[1], x[2]), 
                     elems=(y_true, y_pred, mask), 
                     fn_output_signature=tf.float32)
    return 1. - tf.reduce_mean(vals)

@tf.function
def rmse_for_nsp(y_true, y_pred, mask=None, nsp_label=None, segment_emb=None):
    inp_shp = tf.shape(y_true)
    residuals = tf.square(y_true - y_pred)

    segment_emb = segment_emb*(1.-tf.expand_dims(nsp_label, 1))

    loss_first_50 = (residuals*mask) * segment_emb
    loss_all = (residuals * mask) * tf.expand_dims(nsp_label, 1)
    loss_total = loss_first_50 + loss_all

    N = tf.where(loss_total == 0., 0., 1.)
    N = tf.reduce_sum(N, axis=1)
    
    mse_mean = tf.math.divide_no_nan(tf.reduce_sum(loss_total, 1), N)
    mse_mean = tf.reduce_mean(mse_mean)
    return mse_mean

@tf.function
def rmse_for_delta_gap(y_true, y_pred):
    residuals = tf.square(y_true - y_pred)
    mse_mean = tf.reduce_mean(residuals)
    mse_mean = tf.math.sqrt(mse_mean)
    return mse_mean

@tf.function
def rmse_for_gap(y_true, y_pred, gap_mask):
    residuals = tf.square(y_true - y_pred)
    residuals = residuals * gap_mask

    mse_mean = tf.math.divide_no_nan(
            tf.reduce_sum(residuals, axis=1),
            tf.reduce_sum(gap_mask, axis=1),
        ) 
    mse_mean = tf.reduce_mean(residuals)
    return mse_mean