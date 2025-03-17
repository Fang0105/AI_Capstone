import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


logging.basicConfig(
    filename="./results/training_half_dataset.log",  # Log file path
    level=logging.INFO,  # Set logging level
    format="%(message)s",  # Format log messages
)

logging.info("🔹 Starting Sentiment Analysis Training Process")


# 🔹 Load labeled YouTube comments dataset
df = pd.read_csv("./../data/english_comments_labeled.csv").sample(frac=0.5, random_state=42)

logging.info(f"Class distribution:\n{df['sentiment'].value_counts()}\n--------------------------\n")

# 🔹 Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)


# 🔹 Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

logging.info(f"TF-IDF Feature Shape: {X_train_tfidf.shape}\n--------------------------\n")


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# 🔹 Initialize Logistic Regression Model
clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)

# 🔹 Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring="accuracy")

# 🔹 Print Results
logging.info(f"Cross-Validation Scores: {cv_scores}")
logging.info(f"Mean Accuracy: {np.mean(cv_scores):.4f}")
logging.info(f"Standard Deviation: {np.std(cv_scores):.4f}\n--------------------------\n")


from sklearn.metrics import accuracy_score, classification_report

# 🔹 Train on Full Training Set
clf.fit(X_train_tfidf, y_train)

# 🔹 Predict on Test Set
y_pred = clf.predict(X_test_tfidf)

# 🔹 Evaluate Model
logging.info(f"Final Model Accuracy on Test Set: {accuracy_score(y_test, y_pred)}")
logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])}")

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Generate Learning Curve Data
train_sizes, train_scores, test_scores = learning_curve(clf, X_train_tfidf, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy")

# Compute Mean and Standard Deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 🔹 Plot Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", color="r", label="Training Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.plot(train_sizes, test_mean, "o-", color="b", label="Validation Accuracy")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="b")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve of Logistic Regression")
plt.legend()
plt.savefig("./results/logistic_regression_tfidf_learning_curve_half_dataset.png")


import seaborn as sns
from sklearn.metrics import confusion_matrix

# 🔹 Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 🔹 Define Class Labels (Modify if Needed)
class_names = ["Negative", "Neutral", "Positive"]

# 🔹 Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("./results/logistic_regression_tfidf_confussion_matrix_half_dataset.png")


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# 🔹 Map labels from (-1, 0, 1) to (0, 1, 2)
y_test_mapped = y_test.replace({-1: 0, 0: 1, 1: 2})

# 🔹 Binarize the labels for multi-class ROC-AUC
y_test_bin = label_binarize(y_test_mapped, classes=[0, 1, 2])

# 🔹 Get the probability scores for each class
y_score = clf.predict_proba(X_test_tfidf)

# 🔹 Compute ROC Curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]


for i in range(n_classes):
    if y_test_bin[:, i].sum() == 0:  # If no positive samples exist
        print(f"Skipping class {i} () due to no positive samples.")
        continue
    
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))

colors = ["blue", "green", "red"]
class_labels = ["Negative", "Neutral", "Positive"]

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"ROC curve for {class_labels[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)  # Diagonal reference line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression (TF-IDF)")
plt.legend(loc="lower right")
plt.savefig("./results/logistic_regression_tfidf_roc_curve_half_dataset.png")



import joblib

# Save the model
joblib.dump(clf, "./models/logistic_regression_tfidf_model_half_dataset.pkl")

# Save the vectorizer
joblib.dump(vectorizer, "./models/tfidf_vectorizer_half_dataset.pkl")
