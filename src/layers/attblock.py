import tensorflow as tf

from src.layers.attention import HeadAttentionMulti


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='tanh'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, head_dim, num_heads, mixer_size, 
                 dropout=0.1, m_alpha=-0.5, mask_format='Q', 
                 use_leak=False, temperature=0., **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mixer_size = mixer_size
        self.dropout = dropout
        self.mask_format = mask_format
        self.m_alpha = m_alpha
        self.use_leak = use_leak
        self.temp = temperature
        self.mha = HeadAttentionMulti(self.head_dim, self.num_heads, m_alpha=self.m_alpha, mask_format=mask_format, temperature=self.temp)
        self.ffn = point_wise_feed_forward_network(self.num_heads*self.head_dim, 
                                                   self.mixer_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        if use_leak:
            self.reshape_leak_1 = tf.keras.layers.Dense(self.head_dim*self.num_heads)
            self.reshape_leak_2 = tf.keras.layers.Dense(self.head_dim*self.num_heads)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, x, training, mask=None, return_weights=False):
        attn_output, att_weights, qk_values, (q,k,v) = self.mha(x, training=training, mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        if self.use_leak:
            attn_output = self.reshape_leak_1(x) + attn_output

        attn_output = self.layernorm1(attn_output, training=training)

        ffn_output  = self.ffn(attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output  = self.dropout2(ffn_output, training=training)

        if self.use_leak:
            ffn_output = self.reshape_leak_2(attn_output) + ffn_output

        ffn_output  = self.layernorm2(ffn_output, training=training)

        if return_weights:
            return ffn_output, att_weights, qk_values, (q,k,v)

        return ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "mixer_size": self.mixer_size,
            "dropout": self.dropout,
            "mask_format": self.mask_format,
            "m_alpha": self.m_alpha,
            "use_leak": self.use_leak,
            "mha": serialize_keras_object(self.mha),
            "ffn": serialize_keras_object(self.ffn),
        })
        return config

    @classmethod
    def from_config(cls, config):
        mha_config = config.pop("mha")
        ffn_config = config.pop("ffn")
        mha_config = deserialize_keras_object(mha_config)
        ffn_config = deserialize_keras_object(ffn_config)
        return cls(mha_config, ffn_config, **config)