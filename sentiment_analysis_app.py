import torch
from torchtext.data.utils import get_tokenizer
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
import tkinter as tk
from tkinter import ttk


class MySequenceClassifierCNN(LightningModule):
    def __init__(self, vocab_size, dim_emb, num_filters, filter_sizes, dim_state):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=dim_emb,
                      out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, dim_state)
        self.output = nn.Linear(dim_state, 2)
        self.accuracy = Accuracy(task='multiclass', num_classes=2)

    def forward(self, sequence_batch):
        emb = self.embedding(sequence_batch)
        emb = emb.permute(0, 2, 1)  # Adjust for 1D convolution
        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        pooled_outs = [F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(
            2) for conv_out in conv_outs]
        cat_out = torch.cat(pooled_outs, dim=1)
        fc_out = F.relu(self.fc(cat_out))
        output = self.output(fc_out)
        return output.squeeze(0)

    def loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def training_step(self, batch, batch_index):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.accuracy(outputs, targets)
        self.log('acc', self.accuracy, prog_bar=True)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_index):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        val_acc = self.accuracy(outputs, targets)
        self.log('val_acc', val_acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": val_acc}


class MySequenceClassifierLSTM(LightningModule):
    def __init__(self, vocab_size, dim_emb, dim_state):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.rnn = nn.LSTM(
            input_size=dim_emb, hidden_size=dim_state, num_layers=1, batch_first=True)
        self.output = nn.Linear(dim_state, 2)
        self.accuracy = Accuracy(task='multiclass', num_classes=2)

    def forward(self, sequence_batch):
        emb = self.embedding(sequence_batch)
        _, (h_n, _) = self.rnn(emb)
        output = self.output(h_n[-1])
        return output.squeeze(0)

    def loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def training_step(self, batch, batch_index):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.accuracy(outputs, targets)
        self.log('acc', self.accuracy, prog_bar=True)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_index):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        val_acc = self.accuracy(outputs, targets)
        self.log('val_acc', val_acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": val_acc}


# Load vocabulary
vocab = torch.load('vocab.pth')

# Load pretrained models
model_cnn = MySequenceClassifierCNN(vocab_size=len(
    vocab), dim_emb=32, num_filters=64, filter_sizes=[3, 4, 5], dim_state=64)
model_cnn.load_state_dict(torch.load('model_cnn.pth'))
model_cnn.eval()

model_lstm = MySequenceClassifierLSTM(
    vocab_size=len(vocab), dim_emb=32, dim_state=64)
model_lstm.load_state_dict(torch.load('model_lstm_better.pth'))
model_lstm.eval()

# Tokenizer
tokenizer = get_tokenizer('basic_english')


def sentiment_analysis_cnn(input_text, model):
    input_tokens = list(tokenizer(input_text))
    input_indices = [vocab[token] for token in input_tokens]

    # Ensure minimum sequence length for the CNN model
    filter_sizes = model.convs[0].kernel_size
    min_seq_length = max(max(filter_sizes), 5)
    input_indices.extend([vocab['<pad>']] *
                         (min_seq_length - len(input_indices)))

    input_tensor = torch.tensor(input_indices, dtype=torch.int64).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_class = torch.max(output, dim=-1)
    sentiment = "positive" if predicted_class.item() == 1 else "negative"

    return sentiment


def sentiment_analysis_lstm(input_text, model):
    input_tokens = list(tokenizer(input_text))
    input_indices = [vocab[token] for token in input_tokens]

    # Ensure minimum sequence length for the LSTM model
    min_seq_length = 5
    input_indices.extend([vocab['<pad>']] *
                         (min_seq_length - len(input_indices)))

    input_tensor = torch.tensor(input_indices, dtype=torch.int64).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_class = torch.max(output, dim=-1)
    sentiment = "positive" if predicted_class.item() == 1 else "negative"

    return sentiment


def analyze_sentiment():
    user_input = text_entry.get("1.0", tk.END).strip()

    if user_input.lower() == 'exit':
        root.destroy()
        return

    sentiment_cnn = sentiment_analysis_cnn(user_input, model_cnn)
    sentiment_lstm = sentiment_analysis_lstm(user_input, model_lstm)

    result_label.config(
        text=f"Sentiment (CNN): {sentiment_cnn}\nSentiment (LSTM): {sentiment_lstm}")


# GUI setup
root = tk.Tk()
root.title("Sentiment Analysis App")

# Text entry
text_entry = tk.Text(root, wrap=tk.WORD, width=40, height=5)
text_entry.pack(pady=10)

# Analyze button
analyze_button = ttk.Button(
    root, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack(pady=10)

# Result label
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the GUI
root.mainloop()
