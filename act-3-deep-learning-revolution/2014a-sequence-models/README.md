# Sequence Models (2014)

## The Story

Language. Time series. Music. Machines begin to understand sequences. Vanishing gradients plagued RNNs until LSTM gates solved them. Seq2Seq enables machine translation.

## Key Concepts

- **RNN** — Networks with memory of previous inputs
- **LSTM** — Gates that control what to remember and forget
- **GRU** — A simpler alternative to LSTM
- **Embeddings** — Words as vectors in space

## What You Build

- Character-level language model trained on Shakespeare
- Watch a machine learn syntax, rhythm, even style

## Run

```bash
cd lstm
python lstm_shakespeare.py
```

## Key Insight

Sequence matters. The order of words changes meaning entirely. LSTMs were the best we had — until attention replaced them.
