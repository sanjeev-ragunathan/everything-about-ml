# The First AI Winter (1969–1980s)

> *"Perceptrons have been widely publicized as 'pattern recognition' or 'learning' machines and as such have been a source of one of the liveliest controversies in the recent history of science."* — Minsky & Papert, 1969

## The Story

Minsky & Papert proved perceptrons can't solve XOR — a devastating blow. Funding dried up. ML nearly died. Then in 1986, Rumelhart, Hinton & Williams popularized backpropagation — and everything changed.

## Key Concepts

- **XOR problem** — Why a single layer fails
- **Multi-layer networks** — Adding hidden layers to solve non-linear problems
- **Backpropagation** — How errors flow backward to update weights
- **Gradient descent** — Following the slope downhill to minimize loss

## What You Build

- 2-layer neural network solving XOR
- Hand-coded forward pass, loss, backprop, weight updates
- No NumPy — you feel every gradient

## Run

```bash
cd neural-net-xor
python neural_net.py
```

## Key Insight

Adding just one hidden layer lets the network learn *any* pattern — but training it requires backpropagation, which took decades to figure out.
