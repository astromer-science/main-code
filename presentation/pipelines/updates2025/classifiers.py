import tensorflow as tf
import numpy as np

from src.models.astromer_1 import build_input as build_input_base
from tensorflow.keras import layers, Model
from src.layers.input import GammaWeight

def get_embedding(astromer: Model, inp_placeholder: dict, train_encoder: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Extracts the encoder from the base astromer model and computes the embeddings.

    Args:
        astromer (Model): The base ASTROMER model.
        inp_placeholder (dict): The dictionary of Keras input placeholders.
        train_encoder (bool): If True, the encoder's weights will be trainable.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: A tuple containing the embedding tensor 'z' and the
                                     boolean mask tensor.
    """
    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_encoder
    embedding = encoder(inp_placeholder)
    mask = tf.cast(inp_placeholder['mask_in'], dtype=tf.bool)
    return embedding, mask

def att_avg(astromer: Model, config: dict, train_encoder: bool = False) -> Model:
    """
    Builds a classifier using an attention mechanism followed by average-pooling.

    This function first applies a self-attention layer to the embeddings. This
    allows the model to create a contextualized representation for each token,
    weighing the importance of other tokens in the sequence.

    Then, it uses a GlobalAveragePooling1D layer to aggregate these contextualized
    vectors into a single feature vector. This final vector, which represents
    the entire sequence, is fed into a feed-forward network for classification.

    Args:
        astromer (Model): The pre-trained base model (e.g., ASTROMER) from which
                          the sequence embeddings are extracted.
        config (dict): A configuration dictionary containing model parameters.
                       Must include 'window_size' (int) and 'num_cls' (int).
        train_encoder (bool): If True, the weights of the `astromer` encoder
                              will be unfrozen and trained along with the
                              classifier head. Defaults to False.

    Returns:
        Model: A new `tf.keras.Model` that takes the same inputs as `astromer`
               and outputs classification logits of shape (batch_size, num_cls).
    """
    # 1. Create input placeholders and retrieve embeddings from the base model
    inp_placeholder = build_input_base(config['window_size'])
    z, mask = get_embedding(astromer, inp_placeholder, train_encoder)

    # 2. Attention Mechanism
    # Create the correct mask shape and apply self-attention.
    attention_mask = tf.squeeze(mask, axis=2)
    context_sequence = layers.Attention(name='attention_layer')(
        [z, z], mask=[attention_mask, attention_mask]
    )

    # 3. Masked Average-Pooling
    aggregated_vector = layers.GlobalAveragePooling1D()(context_sequence, mask=attention_mask)

    # 4. Classifier Head
    # The aggregated vector is fed into a standard feed-forward network.
    x = layers.Dense(1024, activation='relu')(aggregated_vector)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)

    y_pred = layers.Dense(config['num_cls'], name='output_layer')(x)

    return CustomClassifier(inputs=inp_placeholder, outputs=y_pred, name='Attention_Pooling_Classifier')

def att_cls(astromer: Model, config: dict, train_encoder: bool = False) -> Model:
    """
    Builds a classifier using the [CLS] token strategy for sequence aggregation.

    This function implements a sophisticated aggregation strategy inspired by
    Transformer models like BERT. Instead of a fixed pooling operation, it
    prepends a special, learnable [CLS] (classification) token to the
    beginning of the input sequence.

    This token is processed along with the rest of the sequence through the
    attention layer. The model is trained to aggregate the entire sequence's
    information into the final output embedding of this [CLS] token. This
    single, learned vector is then extracted and used as the input for the
    final classifier.

    Args:
        astromer (Model): The pre-trained base model (e.g., ASTROMER) from which
                          the sequence embeddings are extracted.
        config (dict): A configuration dictionary containing model parameters.
                       Must include 'embedding_dim' (int), 'window_size' (int),
                       and 'num_cls' (int).
        train_encoder (bool): If True, the weights of the `astromer` encoder
                              will be unfrozen and trained along with the
                              classifier head. Defaults to False.

    Returns:
        Model: A new `tf.keras.Model` that takes the same inputs as `astromer`
               and outputs classification logits.
    """
    inp_placeholder = build_input_base(config['window_size'])
    z, mask = get_embedding(astromer, inp_placeholder, train_encoder)

    # 1. Create and prepend a learnable [CLS] token
    batch_size = tf.shape(z)[0]
    # The [CLS] embedding is a trainable variable
    cls_embedding = tf.Variable(
        initial_value=tf.random.normal(shape=[1, 1, config['embedding_dim']], stddev=0.02),
        trainable=True,
        name='cls_embedding'
    )
    # Tile the [CLS] embedding to match the batch size
    cls_tiled = tf.tile(cls_embedding, [batch_size, 1, 1])

    # Concatenate the [CLS] token at the beginning of the sequence
    z_with_cls = layers.Concatenate(axis=1)([cls_tiled, z])

    # Update the attention mask to account for the new [CLS] token
    cls_mask = tf.zeros((batch_size, 1), dtype=tf.bool)
    attention_mask = tf.squeeze(mask, axis=2)
    mask_with_cls = layers.Concatenate(axis=1)([cls_mask, attention_mask])

    # 2. Apply attention to the full sequence (including [CLS])
    context_sequence = layers.Attention(name='attention_layer')(
        [z_with_cls, z_with_cls], mask=[mask_with_cls, mask_with_cls]
    )

    # 3. Extract the [CLS] token's output vector (at position 0)
    # Instead of pooling, we simply select the first vector from the output sequence.
    context_vector = context_sequence[:, 0, :]

    # 4. Classifier Head
    # The single, aggregated [CLS] vector is fed into the classifier.
    x = layers.Dense(1024, activation='relu')(context_vector)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)

    y_pred = layers.Dense(config['num_cls'], name='output_layer')(x)

    return CustomClassifier(inputs=inp_placeholder, outputs=y_pred, name='CLS_Attention_Classifier')

    
def max_clf(astromer: Model, config: dict, train_encoder: bool = False) -> Model:
    """
    Builds a classifier using max-pooling over the ASTROMER embeddings.

    This function creates a new model that aggregates sequence information from a
    base `astromer` model. It applies masked max-pooling across the time
    dimension of the embeddings. To correctly handle variable-length sequences,
    it first replaces any padded values with negative infinity, ensuring they
    are ignored by the `tf.reduce_max` operation. The resulting feature
    vector is then fed into a feed-forward network for final classification.

    Args:
        astromer (Model): The pre-trained base model (e.g., ASTROMER) from which
                          the sequence embeddings are extracted.
        config (dict): A configuration dictionary containing model parameters.
                       Must include 'window_size' (int) and 'num_cls' (int).
        train_encoder (bool): If True, the weights of the `astromer` encoder
                              will be unfrozen and trained along with the
                              classifier head. Defaults to False.

    Returns:
        Model: A new `tf.keras.Model` that takes the same inputs as `astromer`
               and outputs classification logits of shape (batch_size, num_cls).
    """
    # 1. Create input placeholders and retrieve embeddings from the base model
    inp_placeholder = build_input_base(config['window_size'])
    z, mask = get_embedding(astromer, inp_placeholder, train_encoder)

    # 2. Masked Max-Pooling
    # Replace padded values (where mask is True) with negative infinity.
    # This ensures they are never chosen as the maximum value.
    z_masked = tf.where(mask, -tf.experimental.numpy.inf, z)

    # Reduce the sequence to a single vector by taking the max value
    # across the time dimension (axis=1).
    max_z = tf.reduce_max(z_masked, axis=1)

    # 3. Classifier Head
    # The aggregated vector is fed into a standard feed-forward network.
    x = layers.Dense(1024, activation='relu')(max_z)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)

    y_pred = layers.Dense(config['num_cls'], name='output_layer')(x)

    return CustomClassifier(inputs=inp_placeholder, outputs=y_pred, name='Max_Pooling_Classifier')

def avg_clf(astromer: Model, config: dict, train_encoder: bool = False) -> Model:
    """
    Builds a complete classifier model using masked average pooling and an MLP.

    This function constructs a Keras Model that first extracts a single sequence
    of embeddings from the base `astromer` model. It then computes the masked
    average over the sequence dimension and passes the resulting vector through
    a series of Dense layers for classification.

    Args:
        astromer (Model): The base model from which to extract embeddings.
        config (dict): A configuration dictionary. Must include 'window_size'
                       and 'num_cls'.
        train_encoder (bool): If True, the encoder weights are trainable.

    Returns:
        Model: A new, trainable Keras Model for classification.
    """
    # 1. Define inputs and get embeddings from the base model
    inp_placeholder = build_input_base(config['window_size'])
    # The 'get_embedding' function returns the last hidden state and a boolean mask
    z, mask = get_embedding(astromer, inp_placeholder, train_encoder)

    # 2. Manual Masked Average Pooling
    # Cast the boolean mask to float to use it in calculations
    float_mask = tf.cast(mask, dtype=tf.float32)

    # Apply mask and perform manual average pooling
    x = tf.multiply(z, float_mask)
    # Sum along the sequence axis
    x = tf.reduce_sum(x, 1)
    # Divide by the number of non-padded elements to get the average
    x = tf.math.divide_no_nan(x, tf.reduce_sum(float_mask, 1))

    # 3. MLP Classifier Head
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)
    y_pred = layers.Dense(config['num_cls'], name='output_layer')(x)

    # 4. Build and return the final Keras Model
    return CustomClassifier(inputs=inp_placeholder, outputs=y_pred, name='Average_MLP_Classifier')

def skip_avg_clf(astromer: Model, config: dict, train_encoder: bool = False) -> Model:
    """
    Builds a classifier that weights embeddings from all layers, including the input.

    This function implements a skip-connection-like architecture. It extracts
    not only the hidden states from all encoder layers but also the initial
    input embedding (pre-attention blocks). The input embedding is then prepended
    to the list of hidden states.

    This combined list is processed by first averaging each representation over the
    time dimension, and then feeding the results into a `GammaWeight` layer. This
    custom layer computes a learned weighted sum of all representations (including
    the original input), allowing the model to decide the importance of each
    level of abstraction for the final classification.

    Args:
        astromer (Model): The base model with a custom encoder. The encoder must
                          have an `input_format` method and support the
                          `z_by_layer=True` argument.
        config (dict): Configuration dictionary with 'window_size' and 'num_cls'.
        train_encoder (bool): If True, the encoder weights are trainable.

    Returns:
        Model: A new, trainable Keras Model for classification.
    """
    # 1. Define inputs and get the custom encoder layer
    inp_placeholder = build_input_base(config['window_size'])
    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_encoder

    input_embedding, _ = encoder.input_format(inp_placeholder)
    embedding = encoder(inp_placeholder, z_by_layer=True)
    embedding.insert(0, input_embedding)
    
    x = tf.stack(embedding, axis=0)
    mask = 1.- inp_placeholder['mask_in']
    mask = tf.expand_dims(mask, axis=0)
    x = tf.multiply(x, mask) 
    x = tf.reduce_sum(x, 2)
    x = tf.math.divide_no_nan(x, tf.reduce_sum(mask, 2))

    x = GammaWeight(name='gamma_weight')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)
    y_pred = layers.Dense(config['num_cls'], name='output_layer')(x)

    # 6. Build and return the final Keras Model
    return CustomClassifier(inputs=inp_placeholder, outputs=y_pred, name='Skip_Connection_Classifier')

class CustomClassifier(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        
        return {'y_pred': y_pred, 
                'y_true': y}