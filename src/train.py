# src/train.py

import os
import tensorflow as tf
from src.data_loader import (
    load_config,
    get_datasets,
    get_vectorize_layer,
    vectorize_dataset,
    save_vocabulary,
)
from src.model import build_lstm_model

def train():
    config = load_config()

    # Load and preprocess datasets
    train_data, test_data = get_datasets(config)
    vectorize_layer = get_vectorize_layer(config, train_data)
    save_vocabulary(vectorize_layer, config.get('vocab_path', 'outputs/vocab.txt'))
    train_data = vectorize_dataset(train_data, vectorize_layer)
    test_data = vectorize_dataset(test_data, vectorize_layer)

    # Build model
    model = build_lstm_model(config)

    # Define callbacks
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config['model_save_path'],
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.TensorBoard(log_dir=config['log_dir'])
    ]

    # Train the model
    model.fit(
        train_data,
        epochs=config['epochs'],
        validation_data=test_data,
        callbacks=callbacks
    )

if __name__ == "__main__":
    train()
