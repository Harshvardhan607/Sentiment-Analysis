import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure required NLTK data is downloaded
nltk.download('stopwords', quiet=True)

# 1. Load or Create Dataset
DATA_PATH = "data/reviews.csv"
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def create_sample_data():
    """Generates a sample dataset for demonstration purposes."""
    os.makedirs('data', exist_ok=True)
    data = {
        'review_text': [
            "Absolutely love this product! It works flawlessly.",
            "Terrible quality. Broke after two days of use.",
            "It is okay, nothing special but gets the job done.",
            "Highly recommended, excellent value for money.",
            "Waste of money. Customer service was incredibly rude.",
            "The battery life is amazing and the screen is super bright.",
            "I hate this item, it never works as advertised.",
            "Decent purchase, matches the description well."
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative', 'neutral']
    }
    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)
    print("Created sample dataset at data/reviews.csv")
    return df

if not os.path.exists(DATA_PATH):
    df = create_sample_data()
else:
    df = pd.read_csv(DATA_PATH)

# 2. Text Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.lower() # Convert to lowercase
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned_review'] = df['review_text'].apply(clean_text)

# 3. Feature Extraction (TF-IDF)
X = df['cleaned_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Model Training
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# 5. Evaluation
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix Visualization
os.makedirs('models', exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('models/confusion_matrix.png')
print("Saved confusion matrix plot to models/confusion_matrix.png")

# 6. Save Model and Vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"Model and vectorizer saved to {MODEL_PATH} and {VECTORIZER_PATH}")

# 7. Test on New Data
def predict_sentiment(new_review):
    cleaned = clean_text(new_review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

print("\n--- Testing Custom Input ---")
sample_text = "I am extremely satisfied with this purchase, it exceeded my expectations!"
print(f"Review: '{sample_text}'")
print(f"Predicted Sentiment: {predict_sentiment(sample_text)}")