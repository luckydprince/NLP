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

<h2> Task 3: Evaluation and Verification</h2>

<h4>Performance Comparison</h4>

| Attention Type | Training Loss | Training PPL | Validation Loss | Validation PPL |
|---------------|--------------|--------------|----------------|----------------|
| General Attention | 2.890800 | 18.007704 | 2.629544 | 13.867448 |
| Additive Attention | 2.478727 | 11.926078 | 2.130462 | 8.418754 |

Additive attention consistently outperformed general attention across all evaluation metrics.


<h4>Training Loss Curve</h4>
<img width="575" height="455" alt="image" src="https://github.com/user-attachments/assets/4905348a-6ca5-4e44-b258-7b6e49fcc027" />


<h4>Attention Map Visualization</h4>
<img width="632" height="393" alt="image" src="https://github.com/user-attachments/assets/fb3395e9-6482-4f52-b039-8ea640c523fb" />


<h4>Discussion</h4>
Additive attention consistently achieved lower loss and perplexity,
indicating improved alignment modeling. Attention maps show clearer
token-to-token focus compared to general attention.

<h4>Performance Comparison</h4>

The performance of general and additive attention mechanisms was evaluated
using training loss, validation loss, and perplexity (PPL). Additive
attention achieved lower loss and perplexity values in both training and
validation phases.

This indicates that additive attention provides better alignment modeling
between source and target sequences. The lower validation loss and PPL
also suggest improved generalization compared to general attention.

<h4>Analysis and Discussion</h4>

The experimental results demonstrate that additive attention is more
effective than general attention for Filipino–English translation. Additive
attention consistently achieved lower training and validation loss, as well
as lower perplexity, indicating more accurate and confident predictions.

The improved performance can be attributed to the non-linear scoring
function used in additive attention, which allows richer modeling of
source–target interactions. In contrast, general attention relies on a
simpler dot-product formulation, which may limit expressiveness in
low-resource or complex translation settings.

<h2>Task 4: Machine Translation Web Application</h2>

https://github.com/user-attachments/assets/137c64d7-32b6-4bd6-97c8-64ade74d1b91

A simple web-based machine translation application was developed to
demonstrate the deployment of the trained Filipino–English neural machine
translation model. The application allows users to input a sentence in
Filipino and generates the corresponding English translation.

The frontend of the application was implemented using HTML, CSS, and
JavaScript, providing a user-friendly interface with input and output
text areas. The backend was implemented using Python Flask, which serves
as an interface between the web client and the trained neural machine
translation model.

The translation model is based on an LSTM encoder–decoder architecture
with additive (Bahdanau) attention, which was selected based on its
superior performance compared to general attention in Tasks 2 and 3.
During inference, beam search decoding was applied to improve translation
stability and reduce repetitive outputs without retraining the model.

Upon user submission, the input text is sent to the Flask backend via an
HTTP POST request. The backend processes the input using the trained model
and returns the translated sentence, which is then displayed in the web
interface. While the generated translations may not always be fluent or
accurate, this behavior is expected due to limited training data,
restricted training epochs, and the use of a lightweight LSTM-based
architecture.

Overall, the web application demonstrates an end-to-end machine
translation pipeline, from user interaction to neural model inference,
highlighting the practical deployment of a neural machine translation
system.

<h1>A4 - Do you agree</h1>
This project implements an end-to-end Natural Language Processing (NLP) pipeline based on transformer architectures. It begins with training a lightweight BERT model from scratch using the Masked Language Modeling (MLM) objective, following the original BERT framework. The pretrained encoder is then adapted into a siamese architecture inspired by Sentence-BERT to generate semantically meaningful sentence embeddings.<br><br>
The model is fine-tuned on the SNLI dataset for the Natural Language Inference (NLI) task, predicting entailment, neutral, and contradiction using a Softmax classification objective. Performance is evaluated using standard classification metrics, and limitations are analyzed with proposed improvements.<br><br>
Finally, the trained model is integrated into a responsive Flask-based web application with a Bootstrap interface, allowing users to input a premise and hypothesis and obtain NLI predictions in real time.<br><br>
This assignment demonstrates the full workflow from transformer pretraining to sentence-level inference and deployment.<br>

<h2>A4 – Task 3 </h2>
<h3>Evaluation and Analysis</h3>

<h4>3.1 Experimental Setup</h4>
The Sentence-BERT (SBERT) model was fine-tuned on the SNLI dataset for the Natural Language Inference (NLI) task. A subset of the dataset was used due to computational constraints. The model employed a siamese architecture with shared BERT encoders initialized from the masked language modeling (MLM) pretraining conducted in Task 1.<br><br>

Sentence embeddings were derived using mean pooling over the final transformer layer outputs. The concatenated representation (u,v,∣u-v∣) was passed to a linear classifier for three-way classification (entailment, neutral, contradiction).

<h4>Hyperparameters</h4>

|Parameter|Value |
|---------------|---------------|
|Embedding Size	| 128 |
|Transformer Layers	| 2 |
|Attention Heads	| 4 |
|Max Sequence Length	| 64 |
|Batch Size	| 32 |
|Learning Rate	| 2e-5 |
|Epochs	| 3 |
|Optimizer	| AdamW |
|Loss Function	| CrossEntropy (SoftMaxLoss) |
|Pooling Strategy	| Mean Pooling |
|Device	| Apple MPS |

<h4>3.2 Performance Results</h4>
The classification performance on the validation split is summarized below:

| Class	| precision	recall |	f1-score |	support|
|---------------|---------------|---------------|---------------|
|entailment	| 0.41	| 0.66	| 0.51	| 3329 |
|neutral |	0.43 |	0.31	|0.36	| 3235 |
|contradiction	| 0.37	 0.24	| 0.29	| 3278|
				
|accuracy	| 	|  |	0.41| 	9842|
|macro avg	| 0.40 | 0.40	| 0.39	| 9842 |
|weighted avg	| 0.40	| 0.41	| 0.39	| 9842 | 

The model achieved an overall accuracy of 0.41, exceeding the random baseline of approximately 0.33 for a three-class classification problem. The macro-averaged F1-score of 0.39 indicates moderate performance across classes.<br><br>

The model demonstrated relatively strong recall for the entailment class (0.66), suggesting that the learned embeddings effectively capture positive semantic alignment between sentence pairs. However, performance on the contradiction class was weaker (F1 = 0.29), indicating difficulty in modeling nuanced semantic opposition, the neutral class showed moderate performance but lower recall (0.31), suggesting that fine-grained semantic distinction remain challenging.<br><br>

Overall, the results confirms that Siamese architecture successfully learned semantically meaningful sentence embeddings, although performance remains significantly below state-of-the-art SBERT models trained with full-scale BERT-base architectures.<br><br>

<h4>3.3 Limitations and Challenges</h4>
Despite successful implementation, several limitations impacted performance.

	1. Reduced Model Capacity
The implemented BERT encoder was substantially smaller than the standard BERT-base model. Specifically:

Component	Implemented Model	BERT-base
| Component  | Implemented Model | BERT-base| 
| ------------- | ------------- | ------------- |
| Layers  | 2  | 12  |
| Hidden Size  | 128  | 768  |
| Attention Heads  | 4  | 12  |

This reduced representational capacity limits the model’s ability to capture complex contextual interactions and semantic subtleties required for robust NLI performance.

	2. Simplified Tokenization
A whitespace-based tokenizer was used instead of subword tokenization (e.g., WordPiece). This approach:
	-Lacks handling of out-of-vocabulary words
	-Cannot model morphological variations effectively
	-Reduces semantic granularity

As a result, lexical coverage and representation quality are reduced compared to pretrained BERT tokenization strategies.

	3. Limited Training Data
Due to memory constraints, only a subset of the SNLI dataset was used for training. The full SNLI training set contains over 550,000 examples. Training on a reduced subset limits generalization and contributes to lower overall performance.

	4. Hardware Constraints
Training was performed on Apple Silicon (MPS backend), which imposed constraints on:
	-Batch size
	-Vocabulary size
	-Embedding dimension
	-Transformer depth
These restrictions prevented scaling the architecture toward BERT-base specifications.

	5. Absence of Hyperparameter Optimization
The model was trained using fixed hyperparameters without systematic tuning. No learning rate scheduler, warmup strategy, or early stopping mechanism was applied. This likely limited convergence quality and overall performance.

<h4>3.4 Proposed Improvements</h4>
Several Modifications could substantially improve performance:
	1. Increase Model Depth and Hidden Size
Expanding the transformer layers (e.g., 4-6 layers) and increasing embedding dimension (e.g., 256-768) would enhance contextual modeling capacity.
	2. Adopt Subword Tokenization
Implementing WordPiece or BPE tokenization would improve vocabulary coverage and semantic precision.
	3. Train on Full SNLI or MNLI Dataset
Utilizing the complete dataset, or incorporating the Multi-Genre Natural Language Inference, would improve robustness and cross-domain generalization.
	4. Introduce Learning Rate Scheduling
Applying linear warmup and decay strategies could stabilize training and improve convergence.
	5. Experiment with Pooling Strategies
Comparing mean pooling, max pooling, and CLS-token pooling may yield improved sentence representations.
	6. Apply Regularization Techniques
Incorporating dropout tuning and weight decay could reduce overfitting and improve generalization.

<h4>3.5 Conclusion</h4>
The implemented Sentence-BERT model successfully learned semantically meaningful sentence embeddings using a Siamese architecture and Softmax classification objective. The achieved accuracy of 0.41 demonstrates effective transfer of pretrained contextual representations to the NLI task and confirms that the architecture functions as intended. While performance remains below that of full-scale BERT-based systems, the results validate the correctness of the implementation and highlight the trade-off between computational constraints and model capacity. Future improvements focusing on architectural scaling, tokenization refinement, and hyperparameter optimization are expected to substantially enhance performance.


https://github.com/user-attachments/assets/9fd91c54-5ee8-4c3e-9250-0927fd089b33

<br><br>
<img width="767" height="844" alt="Screenshot 2026-02-15 at 11 02 26 AM" src="https://github.com/user-attachments/assets/d908b267-0d81-4533-bfc2-bce147bc7a1d" />


