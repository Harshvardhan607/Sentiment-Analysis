# Product Review Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project that classifies product reviews into positive, negative, or neutral sentiments using machine learning.

## Overview
This project uses Python, scikit-learn, and NLTK to process textual data, extract features using TF-IDF (Term Frequency-Inverse Document Frequency), and train a Logistic Regression classification model.

## Features
- **Text Preprocessing:** Removes HTML tags, punctuation, and English stopwords.
- **Feature Extraction:** Converts unstructured text to numerical matrices using `TfidfVectorizer`.
- **Modeling:** Trains a fast and highly interpretable `LogisticRegression` model.
- **Evaluation:** Outputs accuracy, a detailed classification report, and generates a confusion matrix.
- **Serialization:** Saves the trained model and vectorizer using `joblib` for future inference without retraining.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/product-sentiment-nlp.git](https://github.com/yourusername/product-sentiment-nlp.git)
   cd product-sentiment-nlp