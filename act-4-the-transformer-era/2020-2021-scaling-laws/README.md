# Scaling Laws (2020–2021)

## Bigger is better... but how much bigger?

BERT and GPT-2 proved that pre-training works. Bigger models do better. So OpenAI decided to push it — **GPT-3**, June 2020:

```
GPT-2:   1.5 billion parameters,  40 billion tokens
GPT-3:   175 billion parameters, 300 billion tokens
```

116x bigger. 96 Transformer decoder layers. embed_size of 12,288. Same architecture as GPT-2 — same architecture you built from scratch. Just massive.

Estimated training cost: **$4.6 million** in compute.

## The completely unexpected: Emergent Abilities

GPT-2 started doing things it was **never trained to do**.

Nobody taught it arithmetic — but it could answer "What is 47 + 83?" Nobody taught it translation — but it could convert English to French. Nobody taught it to code — but describe a function in English and it writes Python.

These abilities **emerged** from scale. They weren't in GPT-2. They appeared when the model got big enough — like a phase transition. Water at 99°C is liquid. At 100°C it becomes something fundamentally different.

So the question became: how big can we go, and how will the growth be?

## Scaling Laws
Emergent abilities were fine and all - but is that enough to then invest millions and billions into training the models? - this is what convinced people to invest:  
**2020 — OpenAI (Kaplan et al.)** discovered that model performance follows **predictable mathematical laws**:

```
Performance improves as a power law of:
  1. Number of parameters (model size)
  2. Amount of training data
  3. Amount of compute used
```

Not randomly. Predictably. You could plot a curve and forecast how good a model would be *before training it*. AI went from research experiment to engineering problem. Companies could now make business decisions: "$10M in compute gets us here, $100M gets us there."

**2022 — DeepMind — Chinchilla paper.** They discovered GPT-3 was actually **undertrained**. It had way too many parameters for the amount of data it saw.

The optimal ratio: roughly **1 parameter : 20 tokens** of training data.

```
GPT-3:      175B parameters, 300B tokens    → ratio 1:1.7  (severely undertrained)
Chinchilla:  70B parameters, 1.4T tokens    → ratio 1:20   (properly trained)
```

Chinchilla — a smaller model — **matched GPT-3's performance** at a fraction of the compute cost.

The lesson: it's not just about being big. It's about the **balance between size and data.**

## The real surprise: Few-shot learning

Remember fine-tuning? Take BERT, add a layer, train on labeled data, get a classifier. GPT-3 showed you could **skip all of that**.

**Zero-shot** — no examples, just ask:

```
Prompt:  "Classify the sentiment: 'This movie was terrible'"
GPT-3:   "Negative"
```

No training. No fine-tuning. No labeled data. It just knows.

**One-shot** — give one example:

```
Prompt:  "Review: 'Best pizza ever!' → Positive
          Review: 'Waited an hour, food was cold' →"
GPT-3:   "Negative"
```

**Few-shot** — give a few examples:

```
Prompt:  "hello → bonjour
          goodbye → au revoir
          cat →"
GPT-3:   "chat"
```

The model learns the **pattern from the prompt itself** — at inference time, not training time. No gradient updates. No weight changes. Just examples in the text.

This is called **in-context learning**. Why does it work? Nobody fully knows to this day. The model was trained to predict the next token. But somewhere in those billions of parameters, it learned to recognize and continue patterns from just a few examples.

## Chain-of-thought

The most powerful prompting technique. Instead of asking for the answer directly, ask the model to **think step by step**.

We tested this on 47 × 83 (correct answer: 3,901):

**Direct prompting** — three runs, wrong every time:

```
Run 1: 4747
Run 2: 3821
Run 3: 3829
```

**Chain-of-thought** ("think step by step") — three runs, correct twice:

```
Run 1: 3901 ✓  (broke into 47×80 + 47×3 = 3760 + 141)
Run 2: 4940    (wrong steps)
Run 3: 3901 ✓  (same correct method)
```

Same model. Same question. The only difference is **how you asked**. The model can't do 47 × 83 in one shot — but when forced to show its work, it breaks the problem into steps it *can* handle.

This is why the 2024 reasoning models (o1) are important — they do chain-of-thought automatically inside the model, without you asking for it.

## What we built

No training in this chapter. Instead, we explored prompt engineering — the art of getting models to do what you want through clever prompting:

```
Zero-shot:         just ask
Few-shot:          give examples, model follows the pattern
Chain-of-thought:  ask the model to reason step by step
```

Each technique is more powerful than the last. And none required changing a single weight.

#### Running models locally

We used **Ollama** to run **Llama 3.2 (1B parameters)** locally on a MacBook M1. No API keys, no cloud, no costs. The model runs entirely on your machine.

For heavier models (7B, 13B, 70B), local inference becomes slow or impossible without serious hardware. That's where cloud APIs and model optimization (Act 6) come in.

---

## Short Notes

**Previous problem:** Pre-training and fine-tuning worked, but fine-tuning still required labeled data and training time for every new task.

**Solution this provided:** Scale unlocked abilities nobody designed. Few-shot learning eliminated fine-tuning for many tasks — just show examples in the prompt. Chain-of-thought dramatically improved reasoning. The discovery that scaling follows predictable laws turned AI from art into engineering.


## My structure for the document
- BERT and GPT-2 proved pre-training works, so bigger models do better.
- GPT-3 - 175 Billion parameters, 96 layers - 300 Billion tokens of text
- completely UNEXPECTED - **Emergent Abilities**
    - they were doing things they weren't even taught.
    - so the question is how big can we go and hjow will the growth be?
- Scaling Laws
    - 2020 - OpenAI - model growth is predictable
    - 2022 - Deepmind - Chinchilla paper - GPT-3 is undertrained.
        - it had very less data for the amounnt of parameters it had.
        - optimal - 1(parameter):20(tokens)
        - Chinchilla model - 70B parameters, 1.4 Trillion tokens - as par to GPT-3 - at fraction of the comoute cost.
    - it's not about just about being big - it's about the balance between size and data.
- The real surprise - *few-shot learning*
    - remember fine-tuning? GPT-3 could skip that.
    - **zero-shot**: no context, just ask it to classify a movie review - it does it!
    - **one-shot**: give it one example of a positive review - then it even classifies a negative review.
    - **few-shot**: give it a few english to french terms - then it translates any text.
    - the model learns the pattern from the prompt itself.
    - this is called - **in-context learning**
    - why does this work ? nobody fully knows till this day.
    - **chain of thoght**

#### What're we even gonna build in this?
- we'll learn how to run models locally using ollama
- llama3.2:1b
- running locally is very heavy for heavier models, hence not a very feasiblie option right now.