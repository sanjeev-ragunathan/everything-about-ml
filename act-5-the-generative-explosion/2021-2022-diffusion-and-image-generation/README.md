# Diffusion & Image Generation (2021–2022)

## The Story

DALL-E, then DALL-E 2, then Stable Diffusion — open-source and runnable on consumer GPUs. The core idea: add noise to images, then learn to reverse the process. Art will never be the same.

## Key Concepts

- **Forward diffusion** — Gradually add noise until the image is pure static
- **Reverse diffusion** — Learn to denoise step by step
- **CLIP** — Connect images and text in the same space
- **Latent space** — Work in compressed representations for efficiency

## What You Build

- Simple diffusion model — forward and reverse process
- Generate images from pure noise

## Run

```bash
cd diffusion
python diffusion.py
```

## Key Insight

Destruction is easy to model. Creation is just learning to reverse destruction.
