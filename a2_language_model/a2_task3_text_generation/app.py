from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------
# Device configuration
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# -----------------------------
# Load vocabulary objects
# (copied from notebook)
# -----------------------------
import pickle

with open("vocab.pkl", "rb") as f:
    char2idx, idx2char = pickle.load(f)

vocab_size = len(char2idx)

# -----------------------------
# Model definition
# -----------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------
# Load trained model
# -----------------------------
embed_size = 128
hidden_size = 256

model = LSTMLanguageModel(vocab_size, embed_size, hidden_size).to(device)
model.load_state_dict(torch.load("harry_potter_lstm.pth", map_location=device))
model.eval()

# -----------------------------
# Text generation function
# -----------------------------
def generate_text(start_text, gen_length=200, temperature=0.8):
    input_seq = torch.tensor(
        [char2idx[c] for c in start_text.lower() if c in char2idx],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    generated_text = start_text

    for _ in range(gen_length):
        output = model(input_seq)
        logits = output / temperature
        probs = F.softmax(logits, dim=1)

        next_char_idx = torch.multinomial(probs, 1).item()
        generated_text += idx2char[next_char_idx]

        input_seq = torch.cat(
            [input_seq[:, 1:], torch.tensor([[next_char_idx]]).to(device)],
            dim=1
        )

    return generated_text

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        generated_text = generate_text(prompt)
    return render_template("index.html", output=generated_text)

if __name__ == "__main__":
    app.run(debug=True)
