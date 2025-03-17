import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
clf_loaded = joblib.load("./models/logistic_regression_tfidf_model.pkl")

# Load the vectorizer
vectorizer_loaded = joblib.load("./models/tfidf_vectorizer.pkl")

df = pd.read_csv("./../data/newjeans_court_reaction_english_labeled.csv")

y_pred = []
y_true = []

for index, row in df.iterrows():
    comment = [row['text']]
    setiment = row['sentiment']
    comment_tfidf = vectorizer_loaded.transform(comment)
    prediction = clf_loaded.predict(comment_tfidf)
    y_pred.append(prediction[0])
    y_true.append(setiment)

# calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
