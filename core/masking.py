import tensorflow as tf

@tf.function
def create_look_ahead_mask(size):
    mask = tf.math.subtract(1.,
                tf.linalg.band_part(tf.ones((size, size)), -1, 0, name='LowerTriangular'),
                name='LookaHeadMask')
    return mask  # (seq_len, seq_len)

@tf.function
def get_padding_mask(steps, lengths):
    ''' Create mask given a tensor and true length '''
    with tf.name_scope("get_padding_mask") as scope:
        lengths_transposed = tf.expand_dims(lengths, 1, name='Lengths')
        range_row = tf.expand_dims(tf.range(0, steps, 1), 0, name='Indices')
        # Use the logical operations to create a mask
        mask = tf.greater(range_row, lengths_transposed)
        return tf.cast(mask, tf.float32, name='LengthMask')

@tf.function
def get_masked(tensor, frac=0.15):
    """ Add [MASK] values to be predicted
    Args:
        tensor : tensor values
        frac (float, optional): percentage for masking [MASK]
    Returns:
        binary tensor: a time-distributed mask
    """
    with tf.name_scope("get_masked") as scope:
        steps = tf.shape(tensor)[0] # time steps
        nmask = tf.multiply(tf.cast(steps, tf.float32), frac)
        nmask = tf.cast(nmask, tf.int32, name='nmask')

        indices = tf.range(steps)
        indices = tf.random.shuffle(indices)
        indices = tf.slice(indices, [0], [nmask])

        mask = tf.reduce_sum(tf.one_hot(indices, steps), 0)
        mask = tf.minimum(mask, tf.ones_like(mask))
        return mask

@tf.function
def set_random(serie_1, mask_1, serie_2, rnd_frac, name='set_random'):
    """ Add Random values in serie_1
    Note that if serie_2 == serie_1 then it replaces the true value
    Args:
        serie_1: current serie
        mask_1 : mask containing the [MASKED]-indices from serie_1
        serie_2: random values to be placed within serie_1
        rnd_frac (float): fraction of [MASKED] to be replaced by random
                          elements from serie_2
    Returns:
        serie_1: serie_1 with random values
    """
    with tf.name_scope(name) as scope:
        nmasked = tf.reduce_sum(mask_1)
        nrandom = tf.multiply(nmasked, rnd_frac, name='mulscalar')
        nrandom = tf.cast(tf.math.ceil(nrandom), tf.int32)

        mask_indices = tf.where(mask_1)
        mask_indices = tf.random.shuffle(mask_indices)
        mask_indices = tf.reshape(mask_indices, [-1])
        mask_indices = tf.slice(mask_indices, [0], [nrandom])

        rand_mask = tf.one_hot(mask_indices, tf.shape(mask_1)[0])
        rand_mask = tf.reduce_sum(rand_mask, 0)
        rand_mask = tf.minimum(rand_mask, tf.ones_like(rand_mask))
        rand_mask = tf.expand_dims(rand_mask, 1)
        rand_mask = tf.tile(rand_mask, [1, tf.shape(serie_2)[-1]])

        len_s1 = tf.minimum(tf.shape(serie_2)[0],
                            tf.shape(rand_mask)[0])

        serie_2 = tf.slice(serie_2, [0,0], [len_s1, -1])

        rand_vals = tf.multiply(serie_2, rand_mask, name='randvalsmul')

        keep_mask = tf.math.floor(tf.math.cos(rand_mask))

        serie_1 = tf.multiply(serie_1, keep_mask, name='seriemul')

        keep_mask = tf.slice(keep_mask, [0,0], [-1,1])
        mask_1  = tf.multiply(mask_1, tf.squeeze(keep_mask), name='maskmul2')
        serie_1 = tf.add(serie_1, rand_vals)

        return serie_1, mask_1

@tf.function
def reshape_mask(mask):
    ''' Reshape Mask to match attention dimensionality '''
    with tf.name_scope("reshape_mask") as scope:
        return mask[:, tf.newaxis, tf.newaxis, :, 0]
