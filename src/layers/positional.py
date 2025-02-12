import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer

def get_angles_prior_knowledge(times, d_model):
    with tf.name_scope("Get_Angles") as scope:
        t_init = tf.math.log(3/24)  # 3 hours in days
        t_end = tf.math.log(3.*365) # 3 years in days
        # Sample in log space
        exponents = tf.linspace(t_init, t_end, d_model)
        # Transform to normal space
        angle_rates = tf.math.divide_no_nan(1.0, tf.exp(exponents))
        # Compute the angular frequency
        angle_rates = 2*3.14159*angle_rates
        # Compute the argument of the trig functions
        angle_rates = times * angle_rates
        return angle_rates

@tf.function
def get_angles_astromer(times, d_model, c=2., base=10000):
    with tf.name_scope("Get_Angles") as scope:
        dim_indices = tf.range(d_model, dtype=tf.float32)
        
        exponent = tf.divide(tf.multiply(c, dim_indices),
                             tf.cast(d_model, tf.float32))
        
        angle_rates = tf.pow(tf.cast(base, dtype=tf.float32), exponent)
        angle_rates = tf.math.reciprocal(angle_rates)
        angle_rates = times * angle_rates
        return angle_rates

def positional_encoding(times, d_model, base=10000, mjd=False, c=2.):
    with tf.name_scope("PosEncoding") as scope:
        if mjd:
            indices = times
        else:
            indices = tf.range(tf.shape(times)[1], dtype=tf.float32)
            indices = tf.expand_dims(indices, 0)
            indices = tf.tile(indices, [tf.shape(times)[0], 1])
            indices = tf.expand_dims(indices, 2)
        
        angle_rads = get_angles_astromer(indices, d_model)
        
        # SIN AND COS
        def fn(x):
            if x[1] % 2 == 0:
                return (tf.sin(x[0]), x[1])
            else:
                return (tf.cos(x[0]), x[1])
        
        x_transpose = tf.transpose(angle_rads, [2,1,0])
        indices = tf.range(0, tf.shape(x_transpose)[0])
        x_transpose = tf.map_fn(lambda x: fn(x),  (x_transpose, indices))[0]
        pos_encoding = tf.transpose(x_transpose, [2, 1, 0])
        return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEncoder2(tf.keras.layers.Layer):
    def call(self, times, d_model, base=10000, mjd=False, c=2.):
        return positional_encoding(times, d_model, base=base, mjd=mjd, c=c)

   
class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, base=1000, c=2, **kwargs):
        super(PositionalEncoder, self).__init__(**kwargs) 
        self.d_model = d_model
        self.base = base
        self.c = tf.cast(c, tf.float32)

    @tf.function
    def call(self, inputs):
        angle_rads = get_angles_astromer(inputs,
                                         self.d_model,
                                         base=self.base,
                                         c=self.c)
        def fn(x):
            if x[1] % 2 == 0:
                return (tf.sin(x[0]), x[1])
            else:
                return (tf.cos(x[0]), x[1])

        x_transpose = tf.transpose(angle_rads, [2,1,0])
        indices = tf.range(0, tf.shape(x_transpose)[0])
        x_transpose = tf.map_fn(lambda x: fn(x),  (x_transpose, indices))[0]
        pos_encoding = tf.transpose(x_transpose, [2, 1, 0])
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "base": self.base,
            "c":self.c,
        })
        return config
