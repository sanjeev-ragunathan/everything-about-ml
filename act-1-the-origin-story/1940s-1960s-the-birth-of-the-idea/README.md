# The Birth of the Idea (1940s–1960s)

## The Biology

Your brain has ~86 billion neurons. Each one does something surprisingly simple:

- It receives signals from other neurons
- Each connection has a **strength** — some signals matter more than others
- The neuron adds up all the incoming signals
- If the total crosses a **threshold** — it fires. If not — silence

That's it. On or off. 1 or 0.

In 1943, **McCulloch & Pitts** turned this into math. They built a model neuron: take inputs, multiply each by a weight (the connection strength), sum them up, check against a threshold. The first mathematical model of the brain.

## Let's build a neuron that handles boolean gates

With just one neuron and the right weights, you can compute:  
We find correct weights and threshold for each of these neurons.

- **AND** — fires only when all inputs are active (w₁=1, w₂=1, threshold=2)
- **OR** — fires when any input is active (w₁=1, w₂=1, threshold=1)
- **NOT** — fires when the input is absent (w₁=-1, threshold=0). The negative weight is an **inhibitory connection** - exactly how some biological neurons suppress others from firing

Three fundamental logic gates. One tiny model. Just by changing the weights and threshold.

## Is manually setting weights feasible?

The McCulloch-Pitts neuron works, but a human has to choose every weight by hand. That's fine for 2 inputs. But what about 400 inputs - like a camera trying to recognise a letter? You can't hand-pick 400 weights.

The neuron needs to **learn**.

That's what Rosenblatt solved in 1957 → `1957-perceptron/`

## But wait.. what about XOR!?

Even with learning, there's a wall. **XOR** - output 1 when inputs are different, 0 when they're the same.

Try to find weights for this. You can't. Nobody can.

Why? Plot the four inputs:

```
x₂
 1 |  ●(1)    ○(0)
   |
 0 |  ○(0)    ●(1)
   +--------------
      0        1    x₁

● = fires (output 1)
○ = silent (output 0)

Try drawing one straight line that separates ● from ○.
You can't.
```

The neuron's equation draws a straight line through this space. Everything on one side fires, everything on the other doesn't. But the 1s sit on opposite diagonal corners — no single line can separate them. This is called **not linearly separable**.

In 1969, **Minsky & Papert** proved this limitation formally. Funding dried up. Researchers abandoned neural networks. Working on this became a career dead end. For 15 years, AI went cold.

This was the **first AI winter**.