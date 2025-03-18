import torch
import torch.nn as nn
from transformers import AutoModel
from torchviz import make_dot

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

# 🔹 Initialize Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = TransformerBERT().to(device)

# 🔹 Create a Dummy Input for Visualization
dummy_input_ids = torch.randint(0, 30522, (1, 50)).to(device)  # Simulate tokenized input (1 sample, 50 tokens)
dummy_attention_mask = torch.ones((1, 50)).to(device)  # Attention mask (all ones)
dummy_labels = torch.tensor([1]).to(device)  # Dummy label for loss computation

# 🔹 Forward Pass
output = model(dummy_input_ids, dummy_attention_mask, dummy_labels)

# 🔹 Generate Model Graph with torchviz
dot = make_dot(output["logits"], params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.format = "png"  # Save as PNG
dot.render("transformer_bert_model")  # Saves as transformer_bert_model.png

print("Model visualization saved as transformer_bert_model.png 🎉")
