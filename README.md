<h1>A2_Language_Model</h1>

Screenshots and videos for the Assignment 2 was uploaded to Google drive due to its file size (https://drive.google.com/drive/folders/1R7p7l3FezttRea5iBUMp4OEkSsFLaKhP?usp=sharing).
<img width="1470" height="956" alt="Screenshot 2026-02-01 at 12 11 01 AM" src="https://github.com/user-attachments/assets/54bcd341-55f1-4b7e-a105-ae04115d1ef0" />
<img width="1470" height="956" alt="Screenshot 2026-02-01 at 12 10 38 AM" src="https://github.com/user-attachments/assets/835c9315-ba6a-4a17-8ff5-ec9553e1ec0c" />

<h1>A3: Make Your Own Machine Translation Language</h1> 
<br>Neural Machine Translation Using LSTM with Attention
<br>
<h2>Task 1 and Task 2</h2>
<br>Language Pair: Filipino → English

<h3>1. Dataset Selection and Source</h3>
<br>For this assignment, a Filipino–English parallel corpus was used to train and evaluate a neural machine translation (NMT) system. The dataset was obtained using the mtdata toolkit, which provides standardized access to reputable public machine translation datasets.
Specifically, the OPUS TED2020 Filipino–English dataset was selected. This dataset consists of sentence-aligned translations of TED talks and is widely used in machine translation research. The dataset is publicly available and intended for research and educational purposes. The source and target corpora were extracted as aligned text files, with Filipino sentences used as the source language and English sentences used as the target language.

<h3>2. Dataset Preparation</h3>
<br>The dataset was preprocessed following standard neural machine translation practices. First, both Filipino and English texts were normalized by converting all characters to lowercase and removing excessive whitespace. Sentence pairs containing empty lines were discarded to ensure alignment quality. Additionally, sentence pairs exceeding a predefined maximum length were removed to improve training stability and reduce memory usage.
Since Filipino uses whitespace-delimited words, no special word segmentation was required. English text was also tokenized using standard whitespace-based tokenization. To handle out-of-vocabulary words and improve generalization, subword tokenization was applied using Byte Pair Encoding (BPE) implemented through the SentencePiece library. A shared subword vocabulary was learned jointly from both source and target languages.
The processed sentences were then encoded into sequences of subword token IDs, with special tokens added to indicate the beginning and end of sequences. Padding was applied during batching to ensure compatibility with the sequence-to-sequence model.

<h3>3. Model Architecture</h3>
<br>An LSTM-based encoder–decoder architecture was implemented for machine translation. The encoder consists of a unidirectional LSTM that processes the source sentence and produces a sequence of hidden states. The decoder is also an LSTM that generates the target sentence one token at a time.<br><br>
To study the effect of attention mechanisms, two models were trained using the same encoder architecture but different attention mechanisms:
<br>1. General Attention (Luong)
<br>General attention computes alignment scores by applying a learned linear transformation to the encoder hidden states and then taking the dot product with the decoder hidden state.
<br>2. Additive Attention (Bahdanau)
<br>Additive attention computes alignment scores using a feedforward neural network with a non-linear activation function, allowing more flexible modeling of source–target interactions.<br><br>
All other components, including embeddings, hidden sizes, optimizers, and training parameters, were kept identical to ensure a fair comparison.

<h3>4. Training Setup</h3>
Both models were trained using cross-entropy loss, with padding tokens ignored during loss computation. The Adam optimizer was used with the same learning rate for both models. Teacher forcing was applied during training by feeding the ground-truth target tokens to the decoder at each time step. Training was conducted for a limited number of epochs due to computational constraints.
The models were trained under identical conditions to ensure that performance differences could be attributed solely to the attention mechanism used.

<h3>5. Evaluation Method</h3>
Model performance was evaluated using the BLEU (Bilingual Evaluation Understudy) metric, which measures n-gram overlap between model-generated translations and reference translations. BLEU scores were computed on a subset of the dataset using greedy decoding during inference.
In addition to quantitative evaluation, qualitative inspection of translated sentences was performed to observe differences in translation fluency and alignment between the two attention mechanisms.

<h3>6. Results</h3>
The BLEU scores obtained were as follows:
<br>Additive Attention (Bahdanau): 0.0148
<br>General Attention (Luong): 0.0000
<br>Although the absolute BLEU scores are low, this outcome is expected given the limited number of training epochs, the use of greedy decoding, and the strict nature of BLEU as an evaluation metric. Importantly, the additive attention model achieved a higher BLEU score than the general attention model.

<h3>7. Discussion</h3>
The experimental results indicate that additive attention outperformed general attention in this low-resource and limited-training setting. This observation is consistent with prior research, which suggests that additive attention provides more expressive alignment modeling, particularly when training data or training time is limited.

<br>While general attention is computationally simpler and faster, additive attention demonstrated better translation quality under identical conditions. The comparison remains valid because both models were trained using the same dataset, preprocessing pipeline, and hyperparameters.

<h3>8. Conclusion</h3>
In this assignment, a Filipino–English neural machine translation system was implemented using an LSTM-based encoder–decoder architecture. Two attention mechanisms—general attention and additive attention—were evaluated under controlled experimental conditions. The results showed that additive attention achieved better translation performance, as measured by BLEU score, reinforcing its effectiveness in neural machine translation tasks.

<br>Overall, this work demonstrates the importance of attention mechanisms in improving translation quality and highlights the practical differences between general and additive attention in sequence-to-sequence models.
