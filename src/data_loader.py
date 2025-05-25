# src/data_loader.py

import tensorflow as tf
import tensorflow_datasets as tfds
import yaml

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_datasets(config):
    (train_data, test_data), _ = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )

    # Shuffle and batch
    train_data = train_data.shuffle(config['buffer_size']).batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)

    return train_data, test_data

def get_vectorize_layer(config, train_data):
    from tensorflow.keras.layers import TextVectorization

    vectorize_layer = TextVectorization(
        max_tokens=config['vocab_size'],
        output_mode='int',
        output_sequence_length=config['sequence_length']
    )

    train_text = train_data.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    return vectorize_layer

def vectorize_dataset(dataset, vectorize_layer):
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    return dataset.map(vectorize_text)
