# Pre-training Explodes (2018–2020)

## The Big Idea

Before this era: train a model from scratch for every task. Sentiment analysis? Train from scratch. Translation? Train from scratch. Question answering? Train from scratch. Each time starting from random weights, needing massive labeled datasets.

**Pre-training** flipped this. Train one massive model on a huge amount of text — let it learn language itself. Grammar, facts, reasoning, common sense. Then **fine-tune** it on your specific task with a small amount of labeled data.

Same concept as transfer learning from ResNet — don't start from zero, start from knowledge. But applied to language at a scale nobody had tried before.

## BERT — Google (October 2018)

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers.

Uses only the **encoder** half of the Transformer. No decoder. BERT isn't trying to generate text — it's trying to **understand** text.

### Embedding
Involves combining **3** matrices: **word** embedding matrix + **positional** embedding matrix + **segment** embedding matrix

### Pre-training tasks

**MLM (Masked Language Modeling)** — randomly hide 15% of words, predict the missing ones:

```
Input:  "The cat [MASK] on the mat"
Target: "The cat sat on the mat"
```

The model uses context from **both directions** — left and right — to predict the masked word. "Sat" makes sense because of "cat" before AND "on the mat" after. That's the **bidirectional** part.

**NSP (Next Sentence Prediction)** — given two sentences, are they actually consecutive?

```
Sentence A: "I went to the store."
Sentence B: "I bought some milk."    → Yes, these follow each other
Sentence B: "The sun is a star."     → No, random
```

This teaches the model to understand relationships between sentences.

Pre-trained on all of English Wikipedia + BookCorpus — 3.3 billion words.

**Great at: understanding.** Classification, sentiment analysis, named entity recognition, question answering.

## GPT — OpenAI (June 2018)

**G**enerative **P**re-trained **T**ransformer.

GPT's depart from "transforming" of text to "generating" of text through Auto-Regressive next-word prediction.  
Auto-Regressive text generation: make prediction - feed them back - predict the next word

Uses only the **decoder** half of the Transformer. No encoder. GPT predicts the next word — left-to-right only:

```
Input:  "The cat sat on"
Target: "the"
```

Never looks at future words. Same idea as the Shakespeare LSTM — but with a Transformer instead of an LSTM.

**Great at: generating.** Text completion, creative writing, conversation. This is what eventually becomes ChatGPT.

**next-token prediction** with large no.of parameters + tokens = translation, summarization, question answering, and more ..  
hence no need to teach specific tasks

**self-supervised** - no need give it labels telling this is the next word for this sequence - giving that for all possible sequences is impractical

### Scaling up

Same architecture. Just bigger:

```
GPT-1 (2018):  12 layers,  embed=768,    12 heads,   117M parameters
GPT-2 (2019):  48 layers,  embed=1600,   25 heads,   1.5B parameters
GPT-3 (2020):  96 layers,  embed=12288,  96 heads,   175B parameters
```

No new inventions. Same Q, K, V. Same attention. Same skip connections. Just more layers, wider layers, more data. The shocking discovery: **just making it bigger made it smarter.** Capabilities emerged from scale that nobody designed in.

GPT-2 was initially withheld — OpenAI called it "too dangerous." It could generate convincing fake news articles. First time an AI model made headlines for being *too good*.

### The fundamental difference

```
BERT: reads BOTH directions → understands text     → "What does this mean?"
GPT:  reads LEFT-TO-RIGHT   → generates text        → "What comes next?"
```

BERT is a reader who sees the whole page. GPT is a writer who only sees what they've written so far.

## T5 — Google (2019) / BART — Facebook (2019)

These use the **full Transformer** — both encoder and decoder.

**T5** (**T**ext-**t**o-**T**ext **T**ransfer **T**ransformer) had an elegant insight: frame **every** task as text in, text out:
`text input -> text output`  
```
Input: “The \ sat on the \”
Output: “\ cat \ mat”
```

```
"sentiment: this movie was terrible"                      → "negative"
"translate English to French: I love cats"                → "J'aime les chats"
"summarize: [500 word article about climate change]"      → "Climate change is accelerating..."
"question: What color is the sky? context: The sky is blue" → "blue"
```

One model. One format. Just change the prefix. No special output layers, no task-specific architectures.

**BART** (**B**i-directional **A**uto-**R**egressive **T**ransformer) (Facebook) — similar idea, encoder-decoder model. Pre-trained by corrupting text (masking, deleting, shuffling sentences) and learning to reconstruct the original. Strong at summarization and generation.

```
Encoder only:          BERT        → understanding
Decoder only:          GPT         → generating
Encoder + Decoder:     T5, BART    → both (text in → text out)
```

What made T5 and BART so portable? -

## Fine-tuning in practice

### The concept

Take a pre-trained model, add a small task-specific layer, train on your data:

```
Pre-trained BERT (understands language)
  → Add one linear layer (2 outputs: positive/negative)
  → Fine-tune on 1000 movie reviews
  → Sentiment classifier
```

The pre-trained weights do most of the work. Fine-tuning just teaches the model your specific task.

### Hugging Face

The library that made this accessible. Loading a pre-trained model is two lines:

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

### Tokenizer

BERT doesn't take raw text. The tokenizer converts text to numbers:

```python
tokenizer("hello, how are you?")
```

```
input_ids:      [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1]
```

- `input_ids` — each word/subword as a number
- `101` = [CLS] (classification token), `102` = [SEP] (separator) — special tokens BERT expects
- `attention_mask` — 1 = real token, 0 = padding (ignore)

### The pipeline: Load → Tokenize → Train

```python
# 1. Load dataset
dataset = load_dataset("imdb")

# 2. Tokenize
tokenizer("I love this movie", padding="max_length", truncation=True, max_length=128)

# 3. Load pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 4. Training loop — Hugging Face models compute loss internally
outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
loss = outputs.loss
loss.backward()
optimizer.step()
```

## The result

```
Training: 1000 movie reviews, 3 epochs
Test Accuracy: 1.0000
```

**100% accuracy** on 500 reviews it had never seen. With just 1000 training examples and 3 epochs of fine-tuning.

Compare this to the Titanic chapter — feature engineering, scaling, comparing four algorithms, 82% accuracy. Here — load a pre-trained model, fine-tune for 3 epochs, perfect score. BERT already understood language. We just taught it "positive vs negative."

This is why pre-training changed everything.

---

## Short Notes

**Previous problem:** Training from scratch for every task required massive labeled datasets and days of compute. Each new task started from zero.

**Solution this provided:** Pre-train once on huge unlabeled text, fine-tune quickly on small labeled data. One base model, many tasks. BERT understands, GPT generates, T5 does both. Scale proved that bigger models get predictably better — setting the stage for GPT-3 and the modern AI boom.