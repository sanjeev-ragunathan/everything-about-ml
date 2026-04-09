# The First AI Winter (1969–1980s)

## The Fallout (1969)

Minsky & Papert proved a single neuron can't solve XOR. The field collapsed. But the answer was hiding in plain sight — **stack neurons in layers**. Everyone suspected this would work. The problem was nobody knew how to train it.

## The Credit Assignment Problem

In a perceptron, training is simple — the output is wrong, you adjust the weights. But in a multi-layer network, the hidden neurons don't produce the final answer. They produce intermediate values.

When the output is wrong, **which hidden neuron is to blame? And by how much?**

This is the credit assignment problem. It went unsolved for nearly two decades.

## Backpropagation (1986)

Rumelhart, Hinton & Williams figured it out. The idea: take the output error and **propagate it backward** through the network, splitting blame based on how much each weight contributed.

```
1. Forward pass   — push inputs through the network, get a prediction
2. Compute error   — how wrong is the output?
3. Propagate back  — send error backward through the weights
4. Update weights  — adjust everything based on its share of the blame
```

But to make this work, they needed to answer: "if I nudge this weight by a tiny amount, how much does the output change?" That question is a **derivative**.

## The Step Function Problem

The perceptron uses a step function — output is hard 0 or hard 1. The derivative of a flat line is **zero**. So when backpropagation asks "how much did this neuron influence the output?", the answer is almost always zero. Error can't flow backward. Learning stops.

We need something smooth.

## Sigmoid — The Activation Function

Sigmoid squashes any number into a smooth curve between 0 and 1:

```
sigmoid(x) = 1 / (1 + e^(-x))

derivative:
sigmoid'(x) = sigmoid(x) × (1 - sigmoid(x))
```

The derivative is never zero in the useful range. Error can always flow. Backpropagation works.

The output is no longer a hard 0 or 1 — it's a **confidence**. A prediction of 0.98 means "98% confident this is a 1."

## The Formulas

**Forward pass:**
```
hidden_output = sigmoid(w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + bias)
final_output  = sigmoid(w₁·h₁ + w₂·h₂ + ... + wₙ·hₙ + bias)
```

**Output error:**
```
output_error = (target - predicted) × predicted × (1 - predicted)
```

**Hidden error** (sum blame from all output neurons, then scale by derivative):
```
hidden_error = (Σ output_errorᵢ × output_weightᵢ) × hidden_out × (1 - hidden_out)
```

**Weight updates:**
```
output_wᵢ  = output_wᵢ  + learning_rate × output_error × hidden_outᵢ
output_b   = output_b   + learning_rate × output_error

hidden_wᵢ  = hidden_wᵢ  + learning_rate × hidden_error × inputᵢ
hidden_b   = hidden_b   + learning_rate × hidden_error
```

## Why Random Weights?

If all weights start at zero, all hidden neurons compute the same thing, get the same error, and update identically. They never differentiate. Two identical neurons are really just one — and one neuron can't solve XOR.

Random initialization **breaks the symmetry**. Each neuron starts with a unique perspective, specializes over time, and learns its own role.

## The Result

```
Input: [0, 0], Target: 0, Predicted: 0.019 ✓
Input: [0, 1], Target: 1, Predicted: 0.984 ✓
Input: [1, 0], Target: 1, Predicted: 0.984 ✓
Input: [1, 1], Target: 0, Predicted: 0.017 ✓
```

XOR solved. The AI winter is over.

## Key Insight

One hidden layer with backpropagation can learn what a single neuron never could. The error flows backward through the weights, each layer gets its share of blame, and the whole network improves together. This is the foundation everything else is built on.