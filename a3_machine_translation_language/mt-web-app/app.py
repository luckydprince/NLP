from flask import Flask, render_template, request, jsonify
import torch
import sentencepiece as spm

# ---- Model imports (copy from notebook) ----
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
        scores = self.v(energy).squeeze(2)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = attention

    def forward(self, input_token, hidden, cell, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        context, _ = self.attention(hidden[-1], encoder_outputs)
        context = context.unsqueeze(1)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

# ---- Load SentencePiece ----
sp = spm.SentencePieceProcessor()
sp.load("model/mt_bpe.model")

VOCAB_SIZE = sp.get_piece_size()
EMBED_SIZE = 256
HIDDEN_SIZE = 512

encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
attention = AdditiveAttention(HIDDEN_SIZE)
decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, attention)
model = Seq2Seq(encoder, decoder)

model.load_state_dict(torch.load("model/model_additive.pt", map_location="cpu"))
model.eval()

# ---- Translation Function ----
import math

def translate_sentence(sentence, max_len=50, beam_size=3):
    model.eval()

    # Encode source sentence
    src_ids = sp.encode(sentence, out_type=int)
    src_tensor = torch.tensor(src_ids).unsqueeze(0)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    bos_id = sp.piece_to_id("<s>")
    eos_id = sp.piece_to_id("</s>")

    # Beam: (tokens, log_prob, hidden, cell)
    beams = [([bos_id], 0.0, hidden, cell)]
    completed = []

    for _ in range(max_len):
        new_beams = []

        for tokens, score, h, c in beams:
            last_token = tokens[-1]

            if last_token == eos_id:
                completed.append((tokens, score))
                continue

            input_token = torch.tensor([last_token])

            with torch.no_grad():
                output, h_new, c_new = model.decoder(
                    input_token, h, c, encoder_outputs
                )

            log_probs = torch.log_softmax(output, dim=1)
            topk_probs, topk_ids = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_token = topk_ids[0, i].item()
                next_score = score + topk_probs[0, i].item()
                new_beams.append((
                    tokens + [next_token],
                    next_score,
                    h_new,
                    c_new
                ))

        # Keep top beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if not beams:
            break

    completed += [(tokens, score) for tokens, score, _, _ in beams]

    # Select best sequence
    best_tokens = max(completed, key=lambda x: x[1])[0]

    # Remove BOS and EOS
    best_tokens = [
        t for t in best_tokens
        if t not in (bos_id, eos_id)
    ]

    return sp.decode(best_tokens)


# ---- Flask App ----
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    text = request.json["text"]
    translation = translate_sentence(text)
    return jsonify({"translation": translation})

if __name__ == "__main__":
    app.run(debug=True)
