import pandas as pd
import re
import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
import logging

logging.basicConfig(
    filename="./results/training.log",  # Log file path
    level=logging.INFO,  # Set logging level
    format="%(message)s",  # Format log messages
)



# Download stopwords
nltk.download('stopwords')

# Convert dataset to DataFrame and use 2000 samples
df = pd.read_csv("./../data/english_comments_labeled.csv")
# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

df["cleaned_review"] = df["text"].apply(clean_text)


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Convert text into a document-term matrix
vectorizer_lda = CountVectorizer(max_features=500, stop_words='english')
X_train_lda = vectorizer_lda.fit_transform(df["cleaned_review"])

# Train LDA model (discover 10 topics)
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(X_train_lda)

# Print the top words per topic
feature_names = vectorizer_lda.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    logging.info(f"Topic {topic_idx + 1}:")
    tem = " ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
    logging.info(tem)
