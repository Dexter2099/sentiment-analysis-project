import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import tensorflow_datasets as tfds
import streamlit as st
from src.data_loader import load_config, load_vocabulary

@st.cache_resource
def load_resources():
    config = load_config()
    vocab = load_vocabulary(config.get('vocab_path', 'outputs/vocab.txt'))
    vectorize_layer = TextVectorization(
        max_tokens=config['vocab_size'],
        output_mode='int',
        output_sequence_length=config['sequence_length'],
    )
    vectorize_layer.set_vocabulary(vocab)
    model = tf.keras.models.load_model(config['model_save_path'])
    return model, vectorize_layer

@st.cache_data
def load_sample_reviews(num_examples=100):
    """Load a subset of IMDb reviews for demonstration."""
    ds = tfds.load('imdb_reviews', split=f'test[:{num_examples}]', as_supervised=True)
    return [text.numpy().decode('utf-8') for text, _ in ds]

def main():
    st.title('IMDb Review Sentiment')
    model, vectorize_layer = load_resources()
    reviews = load_sample_reviews()

    review = st.selectbox('Choose a sample review', reviews)
    if st.button('Predict Sentiment'):
        inputs = tf.constant([review])
        vectorized = vectorize_layer(inputs)
        pred = model.predict(vectorized, verbose=0)[0][0]
        label = 'Positive' if pred >= 0.5 else 'Negative'
        st.markdown(f'**{label} ({pred:.2f})** - {review}')

if __name__ == '__main__':
    main()
