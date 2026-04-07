# Pre-training Explodes (2018–2020)

## The Story

The insight: train on everything, fine-tune on anything. BERT reads bidirectionally. GPT predicts left-to-right. Scale is the new secret weapon. Hugging Face emerges as the hub for all of it.

## Key Concepts

- **BERT** — Masked language modeling, bidirectional
- **GPT** — Autoregressive, left-to-right prediction
- **Pre-training** — Learn general knowledge from massive data
- **Fine-tuning** — Adapt to specific tasks with small data

## What You Build

- Fine-tune BERT for sentiment analysis using Hugging Face
- See a model trained on the internet adapt to your task in minutes

## Run

```bash
cd bert
python bert_finetune.py
```

## Key Insight

Pre-training amortizes the cost of learning. Instead of training from scratch for every task, you learn once and adapt many times.
