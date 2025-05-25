# src/evaluate.py

import tensorflow as tf
from src.data_loader import load_config, get_datasets, get_vectorize_layer, vectorize_dataset

def evaluate():
    config = load_config()

    # Load and preprocess test data
    _, test_data = get_datasets(config)
    vectorize_layer = get_vectorize_layer(config, test_data)  # Use test data just to adapt layer
    test_data = vectorize_dataset(test_data, vectorize_layer)

    # Load the best model
    model = tf.keras.models.load_model(config['model_save_path'])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print(f'\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    evaluate()
