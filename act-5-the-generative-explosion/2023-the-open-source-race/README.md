# The Open-Source Race (2023)

## The Story

GPT-4 handles images and text. Then Meta releases LLaMA — and the floodgates open. Llama 2, Mistral, Falcon follow. A leaked Google memo says "We have no moat." The debate rages: open vs closed. Claude, Gemini, and others enter the arena.

## Key Concepts

- **Open-source LLMs** — Models you can download, inspect, and modify
- **Quantization** — Shrink models to run on consumer hardware
- **LoRA / QLoRA** — Fine-tune efficiently with minimal resources
- **GGUF format** — Optimized model format for local inference

## What You Build

- Download a small open-source model
- Quantize it to run on your MacBook
- Fine-tune on a custom dataset

## Run

```bash
cd local-llm
python local_llm.py
```

## Key Insight

You don't need a data center. A quantized 7B model on a laptop can be surprisingly capable — and you control every aspect of it.
