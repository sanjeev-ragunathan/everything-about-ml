# The First Real CNN (1998)

## The Story

Yann LeCun built a convolutional net that reads handwritten digits for US banks. Convolutions, pooling, shared weights — a revolutionary idea. Mostly ignored for 14 years until GPUs made it practical.

## Key Concepts

- **Convolutions** — Sliding filters that detect local patterns
- **Pooling** — Downsampling to reduce computation
- **Shared weights** — Same filter applied everywhere
- **MNIST** — The "Hello World" of deep learning

## What You Build

- LeNet architecture in PyTorch
- Handwritten digit recognition
- First taste of GPU acceleration (MPS on Apple Silicon)

## Run

```bash
cd lenet
python lenet.py
```

## Key Insight

Convolutions exploit spatial structure — nearby pixels matter more than distant ones. This insight powers all of computer vision.
