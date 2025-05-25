# src/model.py

import tensorflow as tf
import yaml

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_lstm_model(config):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=config['vocab_size'],
            output_dim=config['embedding_dim']
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config['lstm_units'])),
        tf.keras.layers.Dense(config['dense_units'], activation='relu'),
        tf.keras.layers.Dropout(config['dropout_rate']),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        metrics=['accuracy']
    )

    return model
