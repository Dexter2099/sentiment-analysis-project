import tensorflow as tf
import streamlit as st
from imdb import IMDb
from src.data_loader import load_config, get_datasets, get_vectorize_layer

@st.cache_resource
def load_resources():
    config = load_config()
    train_data, _ = get_datasets(config)
    vectorize_layer = get_vectorize_layer(config, train_data)
    model = tf.keras.models.load_model(config['model_save_path'])
    return model, vectorize_layer

def fetch_reviews(title: str):
    ia = IMDb()
    results = ia.search_movie(title)
    if not results:
        return []
    movie = results[0]
    ia.update(movie, info=['reviews'])
    reviews = movie.get('reviews', [])
    return [r.get('content', '') for r in reviews]

def main():
    st.title('IMDb Review Sentiment')
    model, vectorize_layer = load_resources()
    movie_title = st.text_input('Enter a movie title')
    if movie_title:
        with st.spinner('Fetching reviews...'):
            reviews = fetch_reviews(movie_title)
        if not reviews:
            st.warning('No reviews found for this movie.')
        else:
            inputs = tf.constant(reviews)
            vectorized = vectorize_layer(inputs)
            preds = model.predict(vectorized, verbose=0).flatten()
            for rev, pred in zip(reviews, preds):
                label = 'Positive' if pred >= 0.5 else 'Negative'
                st.markdown(f'**{label} ({pred:.2f})** - {rev}')
            pos_ratio = (preds >= 0.5).mean()
            st.write(f'Overall positive ratio: {pos_ratio * 100:.2f}%')

if __name__ == '__main__':
    main()
