import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load Dataset & Train Model (if model not found)
MODEL_FILE = "fake_news_model.pkl"

if not os.path.exists(MODEL_FILE):
    # Sample training data
    train_texts = [
        "Breaking news: Scientists discover cure for cancer!",  # Real
        "Government is hiding secret UFO bases!",  # Fake
        "COVID-19 vaccines approved by WHO.",  # Real
        "Aliens have landed in New York City!"  # Fake
    ]
    train_labels = [1, 0, 1, 0]  # 1 = Real, 0 = Fake

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)

    model = LogisticRegression()
    model.fit(X_train, train_labels)

    # Save the model & vectorizer
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((vectorizer, model), f)

# Load the trained model
with open(MODEL_FILE, "rb") as f:
    vectorizer, model = pickle.load(f)

def predict_fake_news(text):
    """Predicts whether news is Fake or Real."""
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)[0]
    return "Real" if prediction == 1 else "Fake"
