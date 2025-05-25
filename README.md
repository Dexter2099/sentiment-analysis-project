# Sentiment Analysis on IMDb Reviews 

This portfolio project implements a simple binary sentiment classifier using TensorFlow and Keras. The goal is to classify IMDb movie reviews as positive or negative while showcasing basic TensorFlow skills.

## Dataset
- **Source**: [IMDb Reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- **Size**: 50,000 labeled reviews (25K train / 25K test)

## 🔧 Features
- Text vectorization using Keras `TextVectorization`
- LSTM model with embedding and dropout layers
- Modular Python code structure in `src/`
- Config-driven hyperparameters via `configs/config.yaml`
- TensorBoard logging and model checkpoints

## How to Run
```bash
# Train the model
python main.py --mode train

# Evaluate the model
python main.py --mode eval
```

## Results
Using the default configuration (5 epochs, embedding size 64), the model reaches about **86% accuracy** on the test set. The exact numbers may vary but should be in this range after several epochs of training.

These results demonstrate that the model is effectively learning from the dataset and provide a starting point for experimenting with TensorFlow's layers and tooling.

## What the Model Learns
- During training, the model maps frequently occurring words and phrases to sentiment labels, building an internal vocabulary of positive and negative cues.
- The embedding and LSTM layers capture word order and context, allowing the network to distinguish between phrases like *"not good"* and *"good"*.
- By repeatedly seeing labeled reviews, the network learns a representation of movie review language that lets it predict the probability that a new review is positive.
