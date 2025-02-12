import tensorflow as tf



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
        
        # print(rand_mask)
        
        len_s1 = tf.minimum(tf.shape(serie_2)[0],
                            tf.shape(rand_mask)[0])

        serie_2 = tf.slice(serie_2, [0,0], [len_s1, -1])

        rand_vals = tf.multiply(serie_2, rand_mask, name='randvalsmul')

        keep_mask = tf.math.floor(tf.math.cos(rand_mask))

        serie_1 = tf.multiply(serie_1, keep_mask, name='seriemul')
        keep_mask = tf.slice(keep_mask, [0,0], [-1,1])
        # print(keep_mask)
        mask_1  = tf.multiply(mask_1, tf.squeeze(keep_mask), name='maskmul2')
        serie_1 = tf.add(serie_1, rand_vals)

        return serie_1, mask_1

@tf.function
def mask_sample(sample, msk_frac, rnd_frac, same_frac, max_obs):
    '''
    OLD VERSION
    '''
    input_dict = sample.copy()
    
    x = input_dict['input']

    seq_time = tf.slice(x, [0, 0], [-1, 1])
    seq_magn = tf.slice(x, [0, 1], [-1, 1])
    seq_errs = tf.slice(x, [0, 2], [-1, 1])

    # Save the true values
    time_steps = tf.shape(seq_magn)[0]
    orig_magn = seq_magn

    # [MASK] values
    if msk_frac == 1.:
        mask_out = tf.ones(time_steps) * input_dict['mask']
    else:
        mask_out = get_masked(seq_magn, msk_frac)

    # [MASK] -> Identity
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_out,
                                   seq_magn,
                                   same_frac,
                                   name='set_same')

    # [MASK] -> Random
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_in,
                                   tf.random.shuffle(seq_magn),
                                   rnd_frac,
                                   name='set_random')
    if msk_frac == 1.:
        mask_in  =  1.- mask_out

    mask_out = tf.reshape(mask_out, [time_steps, 1])
    mask_in = tf.reshape(mask_in, [time_steps, 1])

    if time_steps < max_obs:
        mask_fill = tf.ones([max_obs - time_steps, 1], dtype=tf.float32)
        mask_out  = tf.concat([mask_out,  1-mask_fill], 0)
        mask_in   = tf.concat([mask_in,     mask_fill], 0)
        seq_magn  = tf.concat([seq_magn,  1-mask_fill], 0)
        seq_time  = tf.concat([seq_time,  1-mask_fill], 0)
        orig_magn = tf.concat([orig_magn, 1-mask_fill], 0)
        input_dict['mask'] =  tf.concat([input_dict['mask'],
                                        1-tf.reshape(mask_fill, [tf.shape(mask_fill)[0]])], 0)

        reshaped_mask = tf.zeros([max_obs - time_steps,
                                  tf.shape(input_dict['input'])[-1]],
                                  dtype=tf.float32)
        input_dict['input'] = tf.concat([input_dict['input'], reshaped_mask], 0)

    input_dict['input_modified'] = seq_magn
    input_dict['mask_in']  = mask_in 
    input_dict['mask_out'] = mask_out
    return input_dict

def mask_dataset(dataset,
                 msk_frac=.5,
                 rnd_frac=.2,
                 same_frac=.2,
                 window_size=None):
    """
    Mask samples per batch following BERT strategy

    Args:
        dataset: A batched tf.Dataset
        msk_frac: observations fraction per light curve that will be masked
        rnd_frac: fraction from masked values to be replaced by random values
        same_frac: fraction from masked values to be replace by same values

    Returns:
        type: tf.Dataset
    """
    assert window_size is not None, 'Masking per sample needs window_size to be specified'
    dataset = dataset.map(lambda x: mask_sample(x,
                                                msk_frac=msk_frac,
                                                rnd_frac=rnd_frac,
                                                same_frac=same_frac,
                                                max_obs=window_size),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    shapes = {'input' :[None, 3],
              'lcid'  :(),
              'length':(),
              'mask'  :[None, ],
              'label' :(),
              'input_modified': [None, None],
              'mask_in': [None, None],
              'mask_out': [None, None],
              'mean_values':[3]}

    return dataset, shapes