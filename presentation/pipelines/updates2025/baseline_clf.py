import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Layer, GRU, LayerNormalization
from src.layers.positional import PositionalEncoder2
from src.models.astromer_1 import build_input as build_input_base
from presentation.pipelines.referee.classifiers import CustomClassifier


class AstromerInputEmbedding(Layer):
    """
    Combina la codificación de tiempo y la proyección de magnitud.
    Basado en la Sección 2.2 del paper de Astromer 1.
    """
    def __init__(self, pe_dim=256, pe_base=10000, pe_c=2, **kwargs):
        super().__init__(**kwargs)

        # Positional encoding
        self.pe_dim  = pe_dim
        self.pe_base = pe_base
        self.pe_c = pe_c
        self.pe = PositionalEncoder2()
        
        # Linear projection for magnitudes
        self.inp_transform = Dense(pe_dim, name="magnitude_proj")

    def call(self, inputs):
        times, magnitudes = inputs
        x_transformed = self.inp_transform(magnitudes)           
        x_pe = self.pe(times, 
                       self.pe_dim, 
                       base=self.pe_base, 
                       mjd=True, 
                       c=self.pe_c)

        return x_transformed + x_pe


def build_supervised_pooling_classifier(config: dict) -> Model:
    """
    Baseline Supervisado 1: Input Embedding -> Pooling -> Clasificador.
    
    Este modelo NO usa el encoder de Astromer. Se entrena de cero.
    """
    # 1. Input Definiton
    inp_placeholder = build_input_base(config['window_size'])
    
    # 2. Input Embedding
    embedding_layer = AstromerInputEmbedding(pe_dim=config['embedding_dim'])
    sequence_embedding = embedding_layer(
        (inp_placeholder['times'], inp_placeholder['input'])
    )

    # 3. Pooling
    mask_float = 1. - inp_placeholder['mask_in']
    x_masked = tf.multiply(sequence_embedding, mask_float)
    x_sum = tf.reduce_sum(x_masked, axis=1)
    mask_sum = tf.reduce_sum(mask_float, axis=1)
    aggregated_vector = tf.math.divide_no_nan(x_sum, mask_sum)

    # 4. Classifier
    x = Dense(1024, activation='relu')(aggregated_vector)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = Dense(config['num_cls'], name='output_layer')(x)

    return CustomClassifier(inputs=inp_placeholder, 
                       outputs=y_pred, 
                       name='Supervised_Pooling_Baseline')


# --- Baseline 2: RNN (GRU) Supervisado ---
def build_supervised_rnn_classifier(config: dict) -> Model:
    """
    Baseline Supervisado 2: Input Embedding -> GRU -> Clasificador.
    
    Este modelo NO usa el encoder de Astromer. Se entrena de cero.
    Usa un GRU como un clasificador de secuencias estándar.
    """
    # 1. Definir Entradas
    inp_placeholder = build_input_base(config['window_size'])
    mask_2d = tf.squeeze(tf.cast(1.-inp_placeholder['mask_in'], tf.bool), axis=-1)

    # 2. Capa de Input Embedding
    embedding_layer = AstromerInputEmbedding(config['embedding_dim'])
    
    #
    sequence_embedding = embedding_layer(
        (inp_placeholder['times'], inp_placeholder['input'])
    )
    print(sequence_embedding)
    rnn_output = GRU(256, return_sequences=False, name='gru_layer')(
        sequence_embedding, 
        mask=mask_2d
    )
    
    # 4. Classifier
    x = Dense(1024, activation='relu')(rnn_output)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = Dense(config['num_cls'], name='output_layer')(x)

    return CustomClassifier(inputs=inp_placeholder, 
                       outputs=y_pred, 
                       name='Supervised_RNN_Baseline')

def build_raw_rnn_classifier(config: dict) -> Model:
    """
    Baseline Supervisado 3: Raw Input (Mags + Times) -> GRU -> Clasificador.
    
    Este modelo NO usa embeddings ni proyecciones iniciales. 
    Concatena magnitud y tiempo y alimenta directamente a la RNN.
    """
    # 1. Definir Entradas
    inp_placeholder = build_input_base(config['window_size'])
    mask_2d = tf.squeeze(tf.cast(1. - inp_placeholder['mask_in'], tf.bool), axis=-1)

    # 2. Concatenación "En Bruto"
    raw_features = tf.concat([inp_placeholder['input'], inp_placeholder['times']], axis=-1)

    # 3. RNN Layer
    rnn_output = GRU(256, return_sequences=False, name='gru_raw_layer')(
        raw_features, 
        mask=mask_2d
    )
    
    # 4. Classifier Head (Mismo MLP que los otros baselines para comparación justa)
    x = Dense(1024, activation='relu')(rnn_output)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = Dense(config['num_cls'], name='output_layer')(x)

    return CustomClassifier(inputs=inp_placeholder, 
                            outputs=y_pred, 
                            name='Raw_RNN_Baseline')