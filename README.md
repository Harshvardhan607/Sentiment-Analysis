# Product Review Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project that classifies product reviews into positive, negative, or neutral sentiments using machine learning and provides an interactive web interface.

## Overview
This project uses Python, scikit-learn, and NLTK to process textual data, extract features using TF-IDF (Term Frequency-Inverse Document Frequency), and train a Logistic Regression classification model. The frontend is powered by Streamlit, offering a clean, real-time dashboard for users to test sentiment predictions.

## Tech Stack
- **Machine Learning:** scikit-learn, NLTK
- **Data Manipulation:** Pandas, NumPy
- **Web Framework:** Streamlit
- **Serialization:** joblib

## Project Structure
```text
product-sentiment-nlp/
│
├── data/                   # Directory for the training dataset (reviews.csv)
├── models/                 # Directory for the serialized model and vectorizer
├── app.py                  # Streamlit frontend application
├── sentiment_model.py      # Core NLP pipeline and model training script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/product-sentiment-nlp.git
   cd product-sentiment-nlp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data:**
   Place your training dataset (`reviews.csv`) inside the `data/` folder. The file must contain at least two columns: `review_text` and `sentiment`.

4. **Train the Model:**
   Generate the machine learning artifacts (model and vectorizer) by running the training script:
   ```bash
   python sentiment_model.py
   ```
   *This will populate the `models/` folder with `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.*

5. **Launch the Dashboard:**
   Start the interactive web application:
   ```bash
   streamlit run app.py
   ```

## Usage
Once the Streamlit server is running, navigate to the provided local URL (typically `http://localhost:8501`) in your web browser. Type or paste any product review into the text area and click **Analyze Sentiment** to see the model's real-time prediction.

## Future Improvements
- Implement Word2Vec or BERT embeddings for deeper semantic understanding.
- Add hyperparameter tuning using GridSearchCV to optimize model accuracy.
- Expand training to a larger dataset (e.g., Amazon Product Reviews) to broaden the model's vocabulary.