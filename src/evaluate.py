# src/evaluate.py

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from src.data_loader import (
    load_config,
    get_datasets,
    vectorize_dataset,
    load_vocabulary,
)

def evaluate():
    config = load_config()

    # Load and preprocess test data
    _, test_data = get_datasets(config)

    # Recreate vectorization layer using the saved vocabulary
    vocab = load_vocabulary(config.get('vocab_path', 'outputs/vocab.txt'))
    vectorize_layer = TextVectorization(
        max_tokens=config['vocab_size'],
        output_mode='int',
        output_sequence_length=config['sequence_length'],
    )
    vectorize_layer.set_vocabulary(vocab)

    test_data = vectorize_dataset(test_data, vectorize_layer)

    # Load the best model
    model = tf.keras.models.load_model(config['model_save_path'])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print(f'\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    evaluate()
