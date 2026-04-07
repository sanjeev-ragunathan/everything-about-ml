# Attention Is All You Need (2017)

## The Story

Google Brain publishes the paper that rewrites everything. No recurrence. No convolution. Just attention — scaled dot-product, multi-head, positional encoding. Every modern AI descends from this paper.

## Key Concepts

- **Self-attention** — Every token looks at every other token
- **Multi-head attention** — Multiple perspectives simultaneously
- **Positional encoding** — How transformers know word order
- **Layer norm** — Stabilize deep transformer stacks

## What You Build

- Transformer from scratch — every component
- Train on a tiny translation task

## Run

```bash
cd transformer
python transformer.py
```

## Key Insight

Attention lets the model decide what's relevant dynamically, rather than relying on fixed-size windows or sequential processing.
