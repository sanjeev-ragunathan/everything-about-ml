# The Architecture Wars (2014–2015)

## The Story

VGGNet goes deeper. GoogLeNet goes wider. ResNet introduces skip connections and goes to 152 layers — solving the vanishing gradient problem for good.

## Key Concepts

- **ResNet** — Skip connections let gradients flow
- **Batch normalization** — Stabilize training
- **Transfer learning** — Reuse knowledge from one task to another

## What You Build

- Fine-tune a pretrained ResNet on a custom image dataset
- Experience how transfer learning collapses training time

## Run

```bash
cd resnet
python resnet_transfer.py
```

## Key Insight

You don't need to train from scratch. Standing on the shoulders of giants is not just allowed — it's the standard.
