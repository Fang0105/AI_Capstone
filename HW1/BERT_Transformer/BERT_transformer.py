import pandas as pd
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(
    filename="./results/training_quarter_dataset.log",  # Log file path
    level=logging.INFO,  # Set logging level
    format="%(message)s",  # Format log messages
)





# 🔹 Load YouTube comments dataset
df = pd.read_csv("./../data/english_comments_labeled.csv").sample(frac=0.25, random_state=42)

# 🔹 Convert sentiment labels (-1 = Negative, 0 = Neutral, 1 = Positive) to sequential classes (0, 1, 2)
label_mapping = {-1: 0, 0: 1, 1: 2}
df["sentiment"] = df["sentiment"].map(label_mapping)

# 🔹 Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)


from transformers import AutoTokenizer

# 🔹 Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 🔹 Convert text into tokenized format
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=50, return_tensors="pt")

# 🔹 Tokenize datasets
X_train_enc = tokenize_function(X_train.tolist())
X_test_enc = tokenize_function(X_test.tolist())

logging.info(f"Example Tokenized Text: {X_train_enc['input_ids'].shape}\n-----------\n")  # (num_samples, max_length)


import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    """PyTorch Dataset for BERT embeddings."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 🔹 Create PyTorch datasets
train_dataset = BERTDataset(X_train_enc, y_train.tolist())
test_dataset = BERTDataset(X_test_enc, y_test.tolist())


import torch.nn as nn
from transformers import AutoModel

class TransformerBERT(nn.Module):
    """Transformer model using BERT embeddings."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=4),
            num_layers=2
        )
        self.fc = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # Get BERT embeddings
        x = self.transformer(x)  # Transformer Encoder
        x = x.mean(dim=1)  # Mean pooling
        x = self.fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.out(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# 🔹 Initialize model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TransformerBERT().to(device)

logging.info("Using device: {device}\n--------------------------\n")


from transformers import Trainer, TrainingArguments

# 🔹 Define Training Arguments
training_args = TrainingArguments(
    output_dir="./bert_transformer_model_quarter_dataset",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs_quarter_dataset",
    logging_steps=10,
    report_to="tensorboard",  # ✅ Enable TensorBoard
    use_mps_device=True  # ✅ Enables Apple GPU acceleration
)

# 🔹 Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 🔹 Train the model
trainer.train()

from sklearn.metrics import accuracy_score, classification_report

# 🔹 Get Predictions
predictions = trainer.predict(test_dataset)
preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()

# 🔹 Compute Accuracy
accuracy = accuracy_score(y_test.tolist(), preds)
logging.info(f"Test Accuracy: {accuracy:.4f}")
logging.info(f"\nClassification Report:\n{classification_report(y_test, preds, target_names=['Negative', 'Neutral', 'Positive'])}\n--------------------------\n")

import seaborn as sns
from sklearn.metrics import confusion_matrix

# 🔹 Compute Confusion Matrix
cm = confusion_matrix(y_test, preds)

# 🔹 Define Class Labels (Modify if Needed)
class_names = ["Negative", "Neutral", "Positive"]


import matplotlib.pyplot as plt
# 🔹 Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("./results/BERT_transformer_confussion_matrix_quarter_dataset.png")





from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# 🔹 Binarize the labels for multi-class ROC computation
y_test_binarized = label_binarize(y_test.tolist(), classes=[0, 1, 2])  # 0: Negative, 1: Neutral, 2: Positive

# 🔹 Get predicted probabilities (Softmax activation assumed in logits)
y_score = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

# 🔹 Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(["Negative", "Neutral", "Positive"]):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc:.2f})')

# 🔹 Plot random chance line
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

# 🔹 Configure plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve (One-vs-Rest) - BERT Transformer')
plt.legend()
plt.grid()
plt.savefig("./results/BERT_transformer_roc_curve_quarter_dataset.png")



torch.save(model.state_dict(), "./models/model_weights_quarter_dataset.pth")

