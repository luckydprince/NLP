from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# Device (Mac MPS safe)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


############################################
# Load Checkpoint
############################################

checkpoint = torch.load("bert_mlm_checkpoint.pt", map_location=device)

vocab = checkpoint["vocab"]
vocab_size = checkpoint["vocab_size"]
embed_size = checkpoint["embed_size"]
layers = checkpoint["layers"]
heads = checkpoint["heads"]
max_len = checkpoint["max_len"]


############################################
# Define BERT Classes
############################################

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = nn.Embedding(max_len, embed_size)
        self.segment = nn.Embedding(2, embed_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        segment_ids = torch.zeros_like(x)
        return self.token(x) + self.position(positions) + self.segment(segment_ids)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.GELU(),
            nn.Linear(embed_size*4, embed_size)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        forward = self.feed_forward(x)
        return self.norm2(x + forward)


class BERT(nn.Module):
    def __init__(self, vocab_size, embed_size, layers, heads):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, embed_size, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads) for _ in range(layers)]
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SentenceBERT(nn.Module):
    def __init__(self, bert_model, embedding_dim):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(embedding_dim * 3, 3)

    def forward(self, s1, s2):
        o1 = self.bert(s1).mean(dim=1)
        o2 = self.bert(s2).mean(dim=1)
        diff = torch.abs(o1 - o2)
        combined = torch.cat([o1, o2, diff], dim=1)
        logits = self.classifier(combined)
        return logits


############################################
# Initialize Model
############################################

bert = BERT(vocab_size, embed_size, layers, heads).to(device)

state_dict = checkpoint["model_state_dict"]
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mlm_head")}
bert.load_state_dict(filtered_state_dict, strict=False)

model = SentenceBERT(bert, embed_size).to(device)
model.load_state_dict(torch.load("sbert_snli_softmax.pt", map_location=device))
model.eval()


############################################
# Tokenizer
############################################

def tokenize(text):
    return text.lower().split()


def encode(text):
    tokens = tokenize(text)[:max_len-2]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    padding = [vocab["[PAD]"]] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long).to(device)


############################################
# Routes
############################################

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        premise = request.form["premise"]
        hypothesis = request.form["hypothesis"]

        s1 = encode(premise)
        s2 = encode(hypothesis)

        ''' with torch.no_grad():
            logits = model(s1, s2)
            pred = torch.argmax(logits, dim=1).item() '''
        
        with torch.no_grad():
            logits = model(s1, s2)
            print("Logits:", logits)   # <-- DEBUG LINE
            pred = torch.argmax(logits, dim=1).item()


        labels = ["Entailment", "Neutral", "Contradiction"]
        prediction = labels[pred]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
