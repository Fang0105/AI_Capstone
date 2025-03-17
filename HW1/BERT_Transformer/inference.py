import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class TransformerBERT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.transformer(x)  # Transformer Encoder
        x = x.mean(dim=1)  # Mean pooling
        x = self.fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.out(x)
        return logits




# 🔹 Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 🔹 Initialize model and load weights
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TransformerBERT().to(device)
model.load_state_dict(torch.load("./models/model_weights.pth", map_location=device))
model.eval()  # ✅ Set model to evaluation mode

def predict(text):
    """Predict sentiment of the given text input."""
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=50, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment_labels = {0: -1, 1: 0, 2: 1}
    return sentiment_labels[predicted_class]

import pandas as pd

# 🔹 Example usage
if __name__ == "__main__":
    # test_text = "This product is amazing! I love it."
    # sentiment = predict(test_text)
    # print(f"Predicted Sentiment: {sentiment}")
    df = pd.read_csv("./../data/newjeans_court_reaction_english_labeled.csv")

    y_pred = []
    y_true = []

    for index, row in df.iterrows():
        comment = row['text']
        setiment = row['sentiment']
        prediction = predict(comment)
        y_true.append(setiment)
        y_pred.append(prediction)

    # calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
