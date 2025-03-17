from textblob import TextBlob
import pandas as pd

# Function to classify sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    # 1 = Positive, 0 = Neutral, -1 = Negative
    if polarity > 0:
        return 1
    elif polarity == 0:
        return 0
    else:
        return -1

df = pd.read_csv("./../data/newjeans_court_reaction_english.csv")

# Apply sentiment analysis
df["sentiment"] = df["text"].apply(get_sentiment)

# count the number of comments with Positive sentiment
positive_comments = df[df["sentiment"] == 1].shape[0]
# count the number of comments with Neutral sentiment
neutral_comments = df[df["sentiment"] == 0].shape[0]
# count the number of comments with Negative sentiment
negative_comments = df[df["sentiment"] == -1].shape[0]

print("Positive Comments:", positive_comments)
print("Neutral Comments:", neutral_comments)
print("Negative Comments:", negative_comments)

# Save labeled dataset
df.to_csv("./../data/newjeans_court_reaction_english_labeled.csv", index=False)

print(df.head())