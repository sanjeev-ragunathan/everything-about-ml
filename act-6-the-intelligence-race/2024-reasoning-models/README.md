# Reasoning Models (2024)

## The Story

OpenAI's o1 shows models can "think" before answering — spending more compute at inference time to reason step-by-step. Chain-of-thought, tree-of-thought, and reasoning traces become the new frontier. Models get smarter, not just bigger.

## Key Concepts

- **Chain-of-thought** — Break complex problems into steps
- **Test-time compute** — Spend more time thinking, get better answers
- **Reasoning traces** — Show your work
- **Self-consistency** — Sample multiple reasoning paths, pick the consensus

## What You Build

- System that decomposes complex problems into reasoning steps
- Compare direct answers vs chain-of-thought accuracy

## Run

```bash
cd chain-of-thought
python chain_of_thought.py
```

## Key Insight

Thinking is a skill you can teach. Giving a model time to reason changes what it's capable of — without changing a single weight.
