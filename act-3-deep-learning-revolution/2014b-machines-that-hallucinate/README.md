# Machines That Hallucinate (2014)

## The Story

Goodfellow invents GANs at 2am after an argument. A generator vs a discriminator — two networks locked in competition. The first time a machine could create realistic images from nothing.

## Key Concepts

- **Generator** — Creates fake data from random noise
- **Discriminator** — Tries to tell real from fake
- **Adversarial training** — Both networks improve by competing
- **Mode collapse** — When the generator gets lazy

## What You Build

- DCGAN generating handwritten digits
- Watch fake digits emerge from random noise

## Run

```bash
cd dcgan
python dcgan.py
```

## Key Insight

Competition drives creativity — even in machines. GANs showed AI could *generate*, not just classify. This opened the door to all generative AI.
