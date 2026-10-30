import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join([word for word in text.lower().split() if word not in stop_words])

# Streamlit User Interface
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊")
st.title("Product Review Analyzer 📊")
st.write("Enter a product review below to analyze its sentiment using Natural Language Processing.")

# User input
review_input = st.text_area("Review Text", height=150, placeholder="Paste a product review here...")

if st.button("Analyze Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            cleaned_text = clean_text(review_input)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            
            # Display results with dynamic styling
            if prediction == 'positive':
                st.success(f"**Sentiment:** Positive 😃")
            elif prediction == 'negative':
                st.error(f"**Sentiment:** Negative 😞")
            else:
                st.info(f"**Sentiment:** Neutral 😐")