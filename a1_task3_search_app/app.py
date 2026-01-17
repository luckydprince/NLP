from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# ===== LOAD EMBEDDINGS =====
# Choose ONE model (recommended: NEG or GloVe)
embeddings = np.load("emb_neg.npy", allow_pickle=True)

# Rebuild vocabulary (must match training)
import nltk, re
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from collections import Counter

for res in ["reuters", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res)

def load_reuters():
    corpus = []
    for fid in reuters.fileids():
        text = reuters.raw(fid).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = word_tokenize(text)
        if len(tokens) > 2:
            corpus.append(tokens)
    return corpus

corpus = load_reuters()[:300]
MIN_COUNT = 5

all_tokens = [w for doc in corpus for w in doc]
word_counts = Counter(all_tokens)
vocabs = [w for w, c in word_counts.items() if c >= MIN_COUNT]
word2index = {w: i for i, w in enumerate(vocabs)}
index2word = {i: w for w, i in word2index.items()}

# ===== SEARCH FUNCTION =====
def search_similar(query, top_k=10):
    if query not in word2index:
        return []

    q_vec = embeddings[word2index[query]]

    scores = np.dot(embeddings, q_vec)

    top_indices = np.argsort(scores)[::-1][1:top_k+1]

    results = [(index2word[i], float(scores[i])) for i in top_indices]
    return results

# ===== ROUTES =====
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form["query"].lower()
        results = search_similar(query)

    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
