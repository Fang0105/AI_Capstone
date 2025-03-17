import pandas as pd
import langdetect
import emoji
import re

def clean_comment(text):
    """Removes emojis and special characters from text."""
    text = emoji.replace_emoji(text, replace="")  # Remove emojis
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

def is_english(text):
    """Detects if a comment is in English."""
    try:
        return langdetect.detect(text) == "en"
    except:
        return False  # Ignore errors (e.g., empty text)
    
comments_df = pd.read_csv("./../data/newjeans_court_reaction.csv")
# based on ["text"] in commetns_df, create a new dataframe with only English comments and remove emojis and special characters
english_comments_df = comments_df[comments_df["text"].apply(is_english)]
english_comments_df["text"] = english_comments_df["text"].apply(clean_comment)
print(english_comments_df.head())
# Save to CSV
english_comments_df.to_csv("./../data/newjeans_court_reaction_english.csv", index=False)
# Show first few comments