# The Birth of the Idea (1940s–1960s)

> *"New Navy Device Learns by Doing"* — The New York Times, 1957

## The Story

McCulloch & Pitts modeled the first mathematical neuron (1943). Turing asked if machines could learn (1950). Rosenblatt built the Perceptron at Cornell — the first machine that learned from its mistakes (1957).

## Key Concepts

- **Weights** — How important is each input?
- **Bias** — How much evidence is needed to fire?
- **Learning rule** — If wrong, nudge weights toward the correct answer
- **Convergence theorem** — If a solution exists, the perceptron will find it

## The Math

```
output = 1  if (w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + bias) > 0
         0  otherwise

Update rule:
  wᵢ = wᵢ + learning_rate × (target - prediction) × xᵢ
```

## What You Build

- AND, OR, NOT gates — learned, not programmed
- Try XOR — watch it fail (sets up the next chapter)

## Run

```bash
cd perceptron
python perceptron.py
```

## Key Insight

The perceptron only learns **linearly separable** problems. XOR isn't one. This limitation nearly killed AI for a decade.
