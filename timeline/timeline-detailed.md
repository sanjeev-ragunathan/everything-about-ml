# Timeline - The Evolution of Machine Learning

> A complete chronological reference from 1943 to today.  
> Organized by when ideas made their breakthrough - not when papers were published.

---

## Act 1 - The Origin Story

### 1943 - The Birth of the Idea

**People:** Warren McCulloch, Walter Pitts  
**Paper:** *A Logical Calculus of the Ideas Immanent in Nervous Activity*

**Concepts:**
- **McCulloch-Pitts neuron** - first mathematical model of a biological neuron
- **Weighted sum** - neuron takes inputs, multiplies by weights, sums them
- **Threshold / activation** - if sum crosses threshold, fires (1); else 0
- **Step function** - the first activation function (binary: 0 or 1)

**Limitations:** weights had to be set manually. No learning.

**Tech:** early computers (ENIAC era). All manual computation.

---

### 1957 - The Perceptron

**People:** Frank Rosenblatt (Cornell)  
**Paper:** *The Perceptron: A Probabilistic Model for Information Storage*

**Concepts:**
- **Perceptron** - first neural network that could learn
- **Perceptron learning rule:** `w = w + lr × error × x`
- **Learning rate (lr)** - controls step size of updates
- **Bias** - adjustable offset added to weighted sum
- Linear decision boundary (single straight line / hyperplane)

**Tech:** Mark I Perceptron (1958) - custom hardware, 400 photocells, potentiometers as weights.

---

### 1969 - The First AI Winter Begins

**People:** Marvin Minsky, Seymour Papert (MIT)  
**Book:** *Perceptrons*

**Concepts:**
- **XOR problem** - single perceptron cannot solve XOR
- **Linear separability** - perceptron can only separate classes with one straight line
- XOR requires a non-linear boundary → needs multiple layers
- Proved perceptrons fundamentally limited

**Impact:** funding dries up, neural network research stalls for ~15 years.

---

### 1986 - The Renaissance

**People:** David Rumelhart, Geoffrey Hinton, Ronald Williams  
**Paper:** *Learning Representations by Back-Propagating Errors*

**Concepts:**
- **Backpropagation** - efficient algorithm to train multi-layer networks
- **Multi-layer perceptron (MLP)** - stacks of perceptrons with hidden layers
- **Chain rule** - mathematical foundation; propagate error backward through layers
- **Hidden layer** - intermediate layer enabling non-linear decision boundaries
- Solved XOR using a 2-layer network

**Activations:**
- **Sigmoid:** `σ(x) = 1 / (1 + e^(-x))` - smooth, differentiable, output in (0,1)
- Replaced step function because its gradient (0 or undefined) couldn't drive learning

**Also 1986:** **RNN (Recurrent Neural Network)** concept introduced - first network with memory.

---

## Act 2 - The Math Age

### 1980s-1990s - Classical ML Rises

**People & algorithms:**
- **Vladimir Vapnik, Corinna Cortes** - **Support Vector Machines (SVM)** (1995)
- **Leo Breiman** - **Random Forests** (2001), bagging
- **Ross Quinlan** - **Decision Trees** (C4.5, 1993)
- **k-Nearest Neighbors (k-NN)** - simple, distance-based classifier

**Concepts:**
- **Feature engineering** - manually crafting input features
- **Feature scaling / normalization** - `StandardScaler` (mean 0, std 1)
- **Train/test split** - proper evaluation methodology
- **Cross-validation** - robust model evaluation
- **Kernel trick** (SVM) - mapping to higher dimensions for non-linear separation
- **Bagging** (Bootstrap Aggregating) - train many models on random subsets, vote

**Tech:** CPUs only. Neural networks abandoned as "slow and impractical."

---

### 1998 - The First Real CNN

**People:** Yann LeCun  
**Paper:** *Gradient-Based Learning Applied to Document Recognition* (LeNet-5)

**Concepts:**
- **Convolutional Neural Network (CNN)** - networks that understand spatial structure
- **Convolution** - slide a small filter over an image to detect patterns
- **Filter / kernel** - small matrix of weights, learned via backpropagation
- **Feature map** - output of convolution, shows where pattern was detected
- **Pooling** (max pooling, average pooling) - downsample while keeping important features
- **Stride** - step size when sliding the filter
- **Padding** - add border pixels to preserve output size
- **Shared weights** - same filter applied everywhere; massive parameter reduction
- **Local connectivity** - each neuron only looks at a small region

**Activations:**
- **Tanh:** `tanh(x)` - squashes to (-1, 1), zero-centered (better than sigmoid for training)

**Architecture of LeNet-5:**
```
Input (32×32) → Conv → Pool → Conv → Pool → FC → FC → Output (10 classes)
~44,000 parameters
```

**Tech:** CPU training, digit recognition (MNIST), used for real US Postal Service checks.

**Dataset:** **MNIST** - 60,000 handwritten digits, 10 classes.

---

## Act 3 - The Deep Learning Revolution

### 2006 - The Quiet Comeback

**People:** Geoffrey Hinton  
**Paper:** *A Fast Learning Algorithm for Deep Belief Nets*

**Concepts:**
- **Deep Belief Networks** - proved deep networks could be trained
- **Layer-wise pre-training** - train one layer at a time
- **"Deep learning"** term gains traction

---

### 2009 - ImageNet

**People:** Fei-Fei Li (Stanford)  
**Dataset:** **ImageNet** - 14 million labeled images across 20,000 categories

**Event:** **ILSVRC** (ImageNet Large Scale Visual Recognition Challenge) launched 2010 - 1.2M images, 1000 classes.

---

### 2012 - The Big Bang Moment

**People:** Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton (University of Toronto)  
**Paper:** *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet)

**Concepts:**
- **AlexNet** - 8 layers, ~60M parameters
- Won ImageNet 2012 with 15.3% error (vs 26.2% best classical ML) - 10.8 percentage point gap
- **Four key innovations:**
  1. **ReLU activation**
  2. **Dropout** - prevents overfitting and co-dependency
  3. **GPU training** - 2× NVIDIA GTX 580
  4. **Data augmentation** - random flips, crops, color shifts

**Activations:**
- **ReLU (Rectified Linear Unit):** `max(0, x)` - derivative = 1 for positive values
- Solved **vanishing gradient problem** (sigmoid max derivative = 0.25; 0.25^8 ≈ 0.00001)

**Concepts introduced:**
- **Vanishing gradient problem** - gradients shrink exponentially in deep networks with sigmoid/tanh
- **Overfitting** - model memorizes training data, fails on new data
- **Dropout** (50%) - randomly deactivate neurons during training
- **Co-dependency** - neurons becoming reliant on specific partners
- **Data augmentation** - artificially expand dataset

**Tech:** GPUs (NVIDIA CUDA) - 2× GTX 580 to train in ~1 week.

---

### 2014–2015 - The Architecture Wars

**People & papers:**
- **Karen Simonyan, Andrew Zisserman** (Oxford) - **VGGNet** (2014)
- **Christian Szegedy et al.** (Google) - **GoogLeNet / Inception** (2014)
- **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun** (Microsoft Research) - **ResNet** (2015)

**Concepts:**
- **VGGNet:** 19 layers, all 3×3 filters, simple uniform design, 138M parameters
- **GoogLeNet / Inception:** 22 layers, parallel filter sizes (1×1, 3×3, 5×5), 5M parameters
- **Inception module** - multiple filter sizes in parallel, concatenate outputs
- **1×1 convolution** - reduces channel dimensions efficiently
- **Degradation problem** - 56-layer networks perform worse than 20-layer (not overfitting)
- **ResNet:** 152 layers, 3.57% ImageNet error (better than human ~5%)
- **Residual block / skip connection:** `output = F(x) + x`
- **Identity mapping** - if layer has nothing to add, outputs zero → input passes through
- **Gradient highway** - backward pass gradient through `+x` is always 1
- **Transfer learning** - take pretrained model, swap last layer, fine-tune

**Concepts introduced:**
- **Batch Normalization (2015)** - Sergey Ioffe, Christian Szegedy
  - Normalizes layer outputs (mean 0, std 1) per mini-batch
  - Learned parameters gamma and beta
  - Solves internal covariate shift
  - Enables deeper networks and higher learning rates

**Tech:** PyTorch/TensorFlow pretrained model hubs emerge.

---

### 2014a - Sequence Models

**People:** Sepp Hochreiter, Jürgen Schmidhuber (invented LSTM in **1997**; mainstream in 2014)

**Papers:**
- 1997: *Long Short-Term Memory*
- 2014: Sutskever, Vinyals, Le - *Sequence to Sequence Learning with Neural Networks* (**Seq2Seq**)
- 2015: Bahdanau, Cho, Bengio - *Neural Machine Translation by Jointly Learning to Align and Translate* (**Attention**)

**Concepts:**
- **RNN (Recurrent Neural Network)** - processes sequences one step at a time, maintains hidden state
- **Hidden state** - memory that carries forward across time steps
- **Vanishing/exploding gradients through time** - long sequences lose information
- **LSTM (Long Short-Term Memory)** - solved vanishing gradient for sequences
- **Cell state** - long-term memory lane (protected, only addition/multiplication)
- **Hidden state** - short-term memory (changes every step)
- **Three gates:**
  - **Forget gate** - what to erase from memory (sigmoid)
  - **Input gate** - what to add to memory (sigmoid + tanh for candidate)
  - **Output gate** - what to output as new hidden state (sigmoid × tanh)
- **GRU (Gated Recurrent Unit)** - simpler alternative to LSTM (Cho et al., 2014)
- **Seq2Seq** - encoder LSTM + decoder LSTM for translation
- **Context vector** - fixed-size vector summarizing input sentence
- **Bottleneck problem** - context vector can't hold meaning of long sentences
- **Bidirectional LSTM** - reads left-to-right AND right-to-left
- **Bahdanau Attention (2015)** - decoder accesses all encoder hidden states with learned weights
- **Attention weights** - softmax over scores, how much to focus on each input

**Concepts introduced:**
- **Word embeddings** - Word2Vec (Mikolov, 2013), GloVe (Pennington, 2014)
- **Character-level language models** - predict next character

**Tech:** Google Translate adopts Seq2Seq in 2016.

---

### 2014b - Machines That Hallucinate

**People:** Ian Goodfellow  
**Paper:** *Generative Adversarial Networks* (2014)

**Concepts:**
- **GAN (Generative Adversarial Network)** - two networks in competition
- **Generator (G)** - takes noise, produces fake images
- **Discriminator (D)** - classifies images as real or fake
- **Adversarial training** - G and D train together, each improving the other
- **Minimax game** - G minimizes, D maximizes the same objective
- **Mode collapse** - generator produces limited variety
- **Training instability** - if one network dominates, training fails

**Loss function:**
- **BCELoss (Binary Cross Entropy)** - for binary real/fake classification

**Evolution:**
- **DCGAN (2015)** - Radford et al. - convolutional GAN, much better images
- **Transposed convolution** (ConvTranspose2d) - upscaling operation, reverse of conv
- **StyleGAN (2019)** - NVIDIA, photorealistic faces

**Activations:**
- **Leaky ReLU** - `max(0.01x, x)` - allows small negative values, used in DCGAN discriminator
- **Tanh** - used in generator output (range -1 to 1)

---

## Act 4 - The Transformer Era

### 2017 - Attention Is All You Need

**People:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain)  
**Paper:** *Attention Is All You Need*

**Concepts:**
- **Transformer** - architecture based entirely on attention, no recurrence
- **Self-attention** - sequence attends to itself
- **Q (Query)** - "what am I looking for?"
- **K (Key)** - "what do I contain?"
- **V (Value)** - "what information do I provide?"
- **Scaled dot-product attention:** `softmax(QK^T / √d_k) × V`
- **Multi-head attention** - multiple parallel attention heads, each learns different relationships
- **Cross-attention** - queries from decoder, keys/values from encoder
- **Masked self-attention** - prevents looking at future tokens (sets future scores to -inf before softmax)
- **Positional encoding** - adds position info since there's no recurrence
- **Encoder layer:** self-attention → Add+Norm → feed-forward → Add+Norm
- **Decoder layer:** masked self-attention → cross-attention → feed-forward (all with Add+Norm)
- **Layer normalization** - normalizes across features within a sample (different from batch norm)
- **Residual connections** - reused from ResNet
- **Parallel processing** - all tokens processed simultaneously (unlike sequential LSTMs)

**Architecture:**
- 6 encoder layers + 6 decoder layers
- embed_size = 512, num_heads = 8, ff_hidden = 2048

**Tech:** TPUs (Google's Tensor Processing Units) begin to dominate for Transformer training.

---

### 2018–2020 - Pre-training Explodes

**Papers & models:**
- **GPT-1** (2018) - Radford et al. (OpenAI) - 117M parameters
- **BERT** (2018) - Devlin et al. (Google) - 340M parameters
- **GPT-2** (2019) - OpenAI - 1.5B parameters
- **T5** (2019) - Raffel et al. (Google) - Text-to-Text Transfer Transformer
- **BART** (2019) - Facebook AI - encoder-decoder with corruption pretraining

**Concepts:**
- **Pre-training** - train once on massive unlabeled text, learn language broadly
- **Fine-tuning** - adapt pretrained model to specific task with small labeled data
- **BERT** - encoder-only, bidirectional, best for understanding
  - **MLM (Masked Language Modeling)** - predict masked words using both directions
  - **NSP (Next Sentence Prediction)** - are two sentences consecutive?
  - **[CLS] token** - aggregates sentence-level representation
  - **[SEP] token** - separates sentences
- **GPT** - decoder-only, autoregressive, best for generation
- **T5** - encoder-decoder, everything is "text-in, text-out" with task prefixes
- **BART** - encoder-decoder, pretrained by corrupting and reconstructing text
- **Tokenization** - WordPiece (BERT), BPE (GPT), SentencePiece (T5)
- **Subword tokenization** - handles unknown words by splitting into pieces
- **Hugging Face Transformers** - library democratized pretrained models

**Tech / Libraries:**
- **Hugging Face** - `transformers`, `datasets`, `tokenizers`
- **PyTorch** becomes dominant in research
- **TensorFlow 2.0** for production

---

### 2020–2021 - Scaling Laws

**People & papers:**
- **Kaplan et al. (OpenAI, 2020)** - *Scaling Laws for Neural Language Models*
- **Brown et al. (OpenAI, 2020)** - *Language Models are Few-Shot Learners* (**GPT-3**)
- **Hoffmann et al. (DeepMind, 2022)** - *Training Compute-Optimal Large Language Models* (**Chinchilla**)

**Concepts:**
- **GPT-3** - 175B parameters, 96 layers, embed_size 12,288, 96 heads
- **Scaling laws** - performance scales predictably with parameters, data, compute (power law)
- **Emergent abilities** - capabilities that appear only at scale (arithmetic, translation, code)
- **Chinchilla scaling** - optimal ratio is 1 parameter : 20 tokens of training data
- GPT-3 was **undertrained** (175B params, only 300B tokens → should have been 3.5T tokens)
- **In-context learning** - learn from examples in the prompt, no weight updates
- **Zero-shot** - just ask the question
- **One-shot** - give one example
- **Few-shot** - give a few examples
- **Chain-of-thought (CoT)** - Wei et al. (2022) - "think step by step" dramatically improves reasoning
- **Prompt engineering** - designing inputs to get better outputs

**Tech:**
- Training GPT-3 estimated at **$4.6M** in compute
- **Microsoft Azure** supercomputer used for GPT-3 training

---

## Act 5 - The Generative Explosion

### 2021–2022 - Diffusion Models

**People & papers:**
- **Ho, Jain, Abbeel (2020)** - *Denoising Diffusion Probabilistic Models (DDPM)*
- **Dhariwal, Nichol (OpenAI, 2021)** - diffusion beats GANs
- **Rombach et al. (2022)** - *High-Resolution Image Synthesis with Latent Diffusion Models* (**Stable Diffusion**)

**Concepts:**
- **Diffusion models** - generate images by iteratively denoising random noise
- **Forward process** - add noise to image step by step until pure noise
- **Reverse process** - learn to denoise, generating images from noise
- **U-Net** - architecture used for the denoising network
- **CLIP** (OpenAI, 2021) - connects text and image embeddings
- **Classifier-free guidance** - steer generation with text prompts
- **Latent diffusion** - operate in compressed latent space (faster, cheaper)
- **Stable Diffusion** - open-source, runs on consumer hardware

**Models:** DALL-E (2021), DALL-E 2 (2022), Midjourney, Stable Diffusion (2022), Imagen

---

### 2022 - ChatGPT, AI Goes Mainstream

**People:** OpenAI  
**Paper:** *Training language models to follow instructions with human feedback* (Ouyang et al., 2022)  
**Release:** November 30, 2022

**Concepts:**
- **Alignment problem** - making models helpful, harmless, honest
- **RLHF (Reinforcement Learning from Human Feedback)** - three steps:
  1. **SFT (Supervised Fine-Tuning)** - fine-tune on ideal conversations
  2. **Reward Model** - trained on human rankings of responses
  3. **RL (PPO)** - optimize model to maximize reward model score
- **PPO (Proximal Policy Optimization)** - Schulman et al. (2017) - the RL algorithm used
- **KL Divergence Penalty** - prevents the model from drifting too far from SFT baseline
- **Reward hacking** - model exploits reward model flaws instead of being helpful
- **Instruction tuning** - specific form of fine-tuning to follow instructions
- **System prompt** - special instructions given to model before user input
- **Constitutional AI** - Anthropic's alternative to RLHF using AI feedback

**Models:**
- **InstructGPT (2022)** - GPT-3 + RLHF, predecessor to ChatGPT
- **ChatGPT (Nov 2022)** - 100M users in 2 months, fastest-growing consumer app ever
- **GPT-4 (Mar 2023)** - multimodal (vision + text)
- **Claude (Anthropic)** - Constitutional AI approach
- **Bard / Gemini (Google)** - response to ChatGPT

---

### 2023 - The Open-Source Race

**People & papers:**
- **Touvron et al. (Meta, Feb 2023)** - *LLaMA: Open and Efficient Foundation Language Models*
- **Meta (Jul 2023)** - **Llama 2** (open for commercial use)
- **Mistral AI (Dec 2023)** - **Mistral 7B** (France)
- **DeepSeek (2024)** - efficient Chinese models
- **Meta (Apr 2024)** - **Llama 3**
- **Meta (Sep 2024)** - **Llama 3.2**

**Concepts:**
- **Open weights** - model weights released publicly
- **Local inference** - run models on your own hardware
- **Quantization** - reduce precision to save memory
  - FP32 → FP16 → INT8 → INT4
  - 4-bit quantization: 7B model fits in ~3.5 GB RAM
  - Minimal accuracy loss (~1-2%)
- **GGUF format** - quantized model format used by llama.cpp
- **LoRA (Low-Rank Adaptation)** - Hu et al. (2021) - freeze base model, train small adapter layers
- **QLoRA** - Dettmers et al. (2023) - quantized base + LoRA for fine-tuning on consumer GPUs
- **PEFT (Parameter-Efficient Fine-Tuning)** - umbrella term for LoRA, adapters, prefix tuning
- **RAG (Retrieval-Augmented Generation)** - Lewis et al. (2020)
  - Retrieve relevant documents before answering
  - Grounds responses in real data
  - Reduces hallucinations
- **Vector databases** - Pinecone, Weaviate, Chroma, FAISS
- **Embeddings** - convert text to vectors for similarity search

**Tech / Tools:**
- **Ollama** - run LLMs locally with one command
- **llama.cpp** - C++ inference engine for quantized LLMs
- **LangChain** - framework for building LLM applications
- **LlamaIndex** - framework for RAG applications

---

## Act 6 - The Intelligence Race (2024+)

### 2024 - Reasoning Models

- **OpenAI o1 (Sep 2024)** - trained to do chain-of-thought internally
- **Test-time compute** - model "thinks" for longer to solve harder problems
- **DeepSeek R1 (2025)** - open-source reasoning model matching o1

### 2024–2025 - Agentic AI

- **Agents** - LLMs that plan, reason, and use tools
- **Function calling / tool use** - model outputs structured calls to APIs
- **ReAct** - reasoning + acting pattern
- **Multi-agent systems** - multiple AIs collaborating
- **Autonomous agents** - AutoGPT, BabyAGI, Devin

### 2025 - Small Models, Big Impact

- **Model distillation** - large model teaches small model
- **Phi-4, Gemma 2, Llama 3.2** - 1B-7B models competitive with larger ones
- **Edge deployment** - LLMs on phones, laptops, IoT

### 2025–2026 - Multimodal Generation

- **GPT-4o, Gemini, Claude 3** - unified text + image + audio + video
- **Sora (OpenAI)** - text-to-video generation
- **Voice cloning and synthesis** - real-time voice interaction

---

## Activation Functions Through History

| Year | Name | Formula | Notes |
|------|------|---------|-------|
| 1943 | Step | 1 if x > 0 else 0 | Binary, not differentiable |
| 1986 | Sigmoid | 1 / (1 + e^-x) | Smooth, vanishing gradient |
| 1998 | Tanh | (e^x - e^-x)/(e^x + e^-x) | Zero-centered, still vanishes |
| 2012 | ReLU | max(0, x) | Solved vanishing, dominant today |
| 2013 | Leaky ReLU | max(0.01x, x) | Fixes "dying ReLU" |
| 2015 | PReLU | max(αx, x), α learned | Kaiming He |
| 2017 | Swish | x × sigmoid(x) | Google Brain |
| 2019 | GELU | x × Φ(x) | Used in BERT, GPT |

---

## Tech & Hardware Evolution

| Era | Tech | Notes |
|-----|------|-------|
| 1943-1970s | CPUs (early) | Manual, slow, limited |
| 1980s-2000s | CPUs (modern) | Classical ML era, limited neural nets |
| 2007+ | **NVIDIA CUDA** | GPUs become programmable for ML |
| 2012 | GTX 580 | AlexNet trained on 2 of these |
| 2016 | **Google TPU v1** | Custom AI chips |
| 2017+ | TPU v2, v3 | Transformer-scale training |
| 2020+ | A100 GPUs | GPT-3 trained on thousands |
| 2022+ | H100 GPUs | Modern LLM training standard |
| 2023+ | Apple Silicon MPS | Local ML on MacBooks |
| 2024+ | Groq LPU, Cerebras | Inference-optimized hardware |

---

## Libraries & Frameworks Timeline

| Year | Library | Purpose |
|------|---------|---------|
| 2007 | **scikit-learn** | Classical ML (SVM, trees, k-NN, etc.) |
| 2008 | **NumPy** | Numerical computing foundation |
| 2008 | **Pandas** | Data manipulation |
| 2002 | **Torch** (Lua) | Original deep learning library |
| 2015 | **TensorFlow 1.x** (Google) | Symbolic DL |
| 2015 | **Keras** | High-level DL API |
| 2016 | **PyTorch** (Facebook) | Dynamic DL, Python-native |
| 2018 | **Hugging Face Transformers** | Pretrained model hub |
| 2019 | **TensorFlow 2.0** | Eager execution like PyTorch |
| 2023 | **Ollama** | Local LLM runtime |
| 2023 | **llama.cpp** | Efficient quantized inference |
| 2023 | **LangChain** | LLM application framework |
| 2023 | **LlamaIndex** | RAG framework |

---

## Key Datasets

| Year | Dataset | Size | Impact |
|------|---------|------|--------|
| 1998 | **MNIST** | 60K digits | Standard benchmark for decades |
| 2009 | **ImageNet** | 14M images, 20K classes | Enabled AlexNet, CNN era |
| 2009 | **ILSVRC** | 1.2M images, 1000 classes | Annual competition (2010-2017) |
| 2009 | **CIFAR-10** | 60K images, 10 classes | Smaller benchmark |
| 2011 | **IMDB Reviews** | 50K reviews | Sentiment analysis standard |
| 2015 | **COCO** | 330K images | Object detection, captioning |
| 2018 | **BooksCorpus + Wikipedia** | ~16 GB text | Used to train BERT |
| 2019 | **WebText** | 40 GB | Used for GPT-2 |
| 2020 | **Common Crawl** | Hundreds of TB | Used for GPT-3, most modern LLMs |
| 2023 | **LAION-5B** | 5B image-text pairs | Stable Diffusion training |

---

## OWASP Top 10 for LLMs (2025)

| ID | Name | Description |
|----|------|-------------|
| LLM01 | **Prompt Injection** | Crafted inputs override instructions (direct or indirect) |
| LLM02 | **Sensitive Information Disclosure** | Model leaks PII, API keys, training data |
| LLM03 | **Supply Chain** | Malicious third-party models, datasets, plugins |
| LLM04 | **Data and Model Poisoning** | Tampered training data corrupts behavior |
| LLM05 | **Improper Output Handling** | App trusts LLM output without validation |
| LLM06 | **Excessive Agency** | Model has too many tools/permissions |
| LLM07 | **System Prompt Leakage** | Attackers extract system prompts |
| LLM08 | **Vector and Embedding Weaknesses** | RAG attacks, embedding inversion |
| LLM09 | **Misinformation** | Confident hallucinations |
| LLM10 | **Unbounded Consumption** | DoS, cost attacks, resource hijacking |

---

## Key People to Know

| Person | Contribution |
|--------|--------------|
| **Geoffrey Hinton** | Backprop (1986), Deep Belief Nets (2006), AlexNet advisor (2012) |
| **Yann LeCun** | CNN, LeNet (1998), Chief AI Scientist Meta |
| **Yoshua Bengio** | Deep learning, attention (2015), Turing Award with Hinton & LeCun |
| **Andrew Ng** | ML popularization, Coursera, Stanford/Google Brain |
| **Fei-Fei Li** | ImageNet (2009), Stanford |
| **Ian Goodfellow** | GANs (2014) |
| **Ilya Sutskever** | AlexNet (2012), Seq2Seq (2014), OpenAI co-founder |
| **Alex Krizhevsky** | AlexNet (2012) |
| **Kaiming He** | ResNet (2015), batch norm work, Mask R-CNN |
| **Sepp Hochreiter** | LSTM (1997) |
| **Ashish Vaswani** | Transformer (2017) lead author |
| **Jacob Devlin** | BERT (2018) |
| **Alec Radford** | GPT series (2018+), OpenAI |
| **Dario & Daniela Amodei** | Anthropic co-founders, Constitutional AI |
| **Andrej Karpathy** | CS231n, OpenAI, Tesla, nanoGPT educator |

---

*For detailed explanations of each concept, see the corresponding chapter README in the repository.*