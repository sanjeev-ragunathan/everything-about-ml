# The First Real CNN (1998)

## Neural nets came back - but this time for images

The mathematicians and statisticians had tabular data covered. Decision trees, SVMs, random forests — they were fast, elegant, and good enough for spreadsheets. But images were a different beast.

You *could* do image classification with classical ML. Extract hand-crafted features — count the loops, measure the angles, find the curves — and feed them to an SVM. People tried this. It kind of worked. But every new style of handwriting broke the system. The features were brittle because a human had to choose them.

What if the network could learn its own features?

## The problem with regular neural networks on images

Think about feeding an image to the neural net you built in Act 1. A 28×28 image has 784 pixels. You'd flatten them into a row of 784 numbers and connect every pixel to every hidden neuron.

But here's what you lose: **spatial relationships**. Pixel (0,0) is right next to pixel (0,1) — they're neighbors in the image. But in a flat row, the network has no idea they're related. It treats every pixel as independent. It doesn't know what's next to what.

An image isn't made up of single pixels in isolation — it's a **combination** of pixels. A "7" is a horizontal stroke connected to a diagonal stroke. You need to see the neighborhood to see the pattern.

## LeCun's insight: Convolution

**Yann LeCun** (1998) solved this by borrowing from biology again. The visual cortex has cells that each look at a **tiny region** of your visual field — detecting local patterns like edges and corners. These local patterns combine into larger patterns, which combine into objects.

A **convolution** does the same thing. A small filter (say 5×5) slides across the image, looking at one patch at a time. At each position, it multiplies the filter values by the pixel values and sums them up. The result is a number — high where the pattern matches, low where it doesn't.

```
Image patch:     Filter:         
1 0 0            1 0 0
0 1 0      ×     0 1 0     →  (1+1+1) = 3  ← strong match
0 0 1            0 0 1
```

This is how convolution solves the neighborhood problem. The filter sees a 5×5 patch — **every pixel with its neighbors, together**. As it slides across the image, it detects that pattern everywhere. One filter might learn horizontal edges, another vertical edges, another curves.

The filter values are just **weights** — and they're learned through backpropagation, just like everything else. They start random, and the network figures out what patterns are useful for the task.

## Pooling

After convolution, you have a feature map showing *where* and *how strongly* a pattern was found. But do you care about the exact pixel location? If a horizontal edge is at pixel (10,14) vs (11,14), that doesn't change whether the digit is a "7."

**Max pooling** shrinks the map by keeping only the strongest signal in each small region (2×2):

```
0.8  0.2
0.1  0.5   →   0.8  (keep the max)
```

Two benefits: reduces data for the next layer, and makes the network less sensitive to exact position.

## The full LeNet structure

```
Input (1 channel, 28×28)
  → Conv (6 filters, 5×5) → tanh → Pool (2×2)
  → Conv (16 filters, 5×5) → tanh → Pool (2×2)
  → Flatten
  → Fully connected (120) → tanh
  → Fully connected (84) → tanh
  → Output (10 digits)
```

First conv layer learns **simple patterns** — edges, lines. Second conv layer combines those into **complex patterns** — curves, corners, shapes. Fully connected layers make the final decision. Output has 10 neurons, one per digit — highest score wins.

Why more filters in the second layer (16 vs 6)? Simple patterns are few — edges only go so many directions. But combinations of patterns are many. The variety grows as you go deeper.

## Tracking the shapes

```
Input:              28×28 (1 channel)
After Conv1 (5×5):  24×24 (6 channels)   ← 28-5+1 = 24
After Pool1 (2×2):  12×12 (6 channels)   ← 24/2 = 12
After Conv2 (5×5):   8×8  (16 channels)  ← 12-5+1 = 8
After Pool2 (2×2):   4×4  (16 channels)  ← 8/2 = 4
Flatten:            256                   ← 16 × 4 × 4
```

28×28 = 784 pixels compressed down to 256 high-level features. Those 256 numbers aren't pixels anymore — they're a description of *what patterns exist in the image*.

## Enter PyTorch

**PyTorch is to deep learning what Scikit-learn is to classical ML.** But where Scikit-learn hides everything behind `fit` and `predict`, PyTorch gives you more control.

### Tensors

PyTorch's version of arrays. Like NumPy arrays, but with two superpowers:

- **GPU support** — move them to the GPU with `.to("mps")` (Apple Silicon) or `.to("cuda")` (NVIDIA) and all math runs in parallel
- **Gradient tracking** — PyTorch remembers every operation done on a tensor, so it can compute derivatives automatically during backpropagation

```python
x = torch.tensor([1.0, 2.0, 3.0])  # a simple tensor
```

When you feed an image through the network, it's a tensor. The weights are tensors. The output is a tensor. Everything is tensors.

### nn.Module

The base class for every neural network in PyTorch. When you write `class LeNet(nn.Module)`, your class inherits all of PyTorch's machinery — gradient tracking, parameter management, GPU movement.

`super().__init__()` in your `__init__` calls the parent class setup. Without it, PyTorch doesn't know your layers exist and can't track their weights.

You define two methods:
- `__init__` — declare what layers the network has
- `forward` — define how data flows through those layers

You never write a `backward` method. PyTorch builds it automatically from your `forward`.

### Building blocks

**`nn.Conv2d(in_channels, out_channels, kernel_size)`** — a convolution layer.
```python
nn.Conv2d(1, 6, 5)
#         │  │  └── 5×5 filter size
#         │  └───── 6 filters (produces 6 feature maps)
#         └──────── 1 input channel (grayscale image, RGB would be 3)
```
Each filter is a 5×5 grid of learnable weights. 6 filters = 6 different pattern detectors. The layer has `6 × (1 × 5 × 5 + 1) = 156` trainable parameters (weights + biases).

**`nn.MaxPool2d(kernel_size)`** — max pooling.
```python
nn.MaxPool2d(2)
#            └── 2×2 window, keeps the max value from each patch
```
No learnable parameters — it's just a shrinking operation. Halves the width and height.

**`nn.Linear(in_features, out_features)`** — a fully connected layer. The same thing you built by hand in Act 1.
```python
nn.Linear(256, 120)
#         │    └── 120 output neurons
#         └─────── 256 inputs (flattened feature maps)
```
Every input connects to every output. Has `256 × 120 + 120 = 30,840` trainable parameters (weights + biases).

### Loading data

**`transforms.ToTensor()`** — takes a PIL image (pixels 0–255) and converts it to a PyTorch tensor (values 0.0–1.0). The scaling to 0–1 helps training because it keeps values small — same reason we used small random weights.

**`datasets.MNIST(root, train, download, transform)`** — downloads the MNIST dataset (60,000 training images, 10,000 test images of handwritten digits) and applies the transform to each image.

**`DataLoader(dataset, batch_size, shuffle)`** — feeds data to the network in **batches** instead of one image at a time.
```python
DataLoader(train_data, batch_size=64, shuffle=True)
#                      │               └── randomize order each epoch (prevents memorizing sequence)
#                      └────────────────── process 64 images at once
```
Why batches? One image at a time is noisy — one weird image can push weights in a bad direction. The full dataset at once is smooth but slow. Batches of 64 are a middle ground: the GPU processes all 64 in parallel, and the average error across 64 images gives a stable update. This is **mini-batch gradient descent**.

### The training loop

PyTorch doesn't have Scikit-learn's `fit`. You write the loop yourself — but each step is one line:

```python
optimizer.zero_grad()            # reset gradients (otherwise they accumulate across batches)
output = model(images)           # forward pass
loss = loss_fn(output, targets)  # compute error
loss.backward()                  # backpropagation — computes all gradients automatically
optimizer.step()                 # update all weights using the computed gradients
```

**`optimizer.zero_grad()`** — PyTorch *accumulates* gradients by default. If you don't zero them, each batch's gradients add to the previous batch's. That's occasionally useful, but usually a bug. Zero them before each batch.

**`loss_fn = nn.CrossEntropyLoss()`** — the loss function for multi-class classification. It takes the raw output scores (10 numbers) and the correct label (one digit), and computes how wrong the prediction is. It combines softmax (turn scores into probabilities) and negative log likelihood (penalize wrong answers) into one step.

**`loss.backward()`** — this is the magic line. PyTorch traced every operation in the forward pass — every convolution, every tanh, every matrix multiplication. Now it walks backward through that entire chain, computing the derivative at every step. The same backpropagation you hand-coded in Act 1, but through convolution layers, pooling, and everything else. Automatically.

**`optimizer = optim.SGD(model.parameters(), lr=0.01)`** — Stochastic Gradient Descent. `model.parameters()` gives it every learnable weight in the network. After `loss.backward()` computes the gradients, `optimizer.step()` does the familiar update: `weight = weight - learning_rate × gradient`.

### Testing

```python
with torch.no_grad():
```
During testing, you don't need gradients — you're just measuring accuracy, not training. `torch.no_grad()` tells PyTorch to skip gradient tracking, saving memory and speed.

```python
_, predicted = torch.max(outputs, 1)
```
`outputs` is a tensor where each row has 10 scores (one per digit). `torch.max(outputs, 1)` returns two things: the highest score and its index (the predicted digit). The `_` means "I don't care about the score, just give me the index."

## PyTorch building blocks → LeNet layers

Putting it all together — each line in `__init__` maps to a step in the architecture:

```python
nn.Conv2d(1, 6, 5)       # 1 input channel → 6 feature maps, 5×5 filters
nn.MaxPool2d(2)           # shrink by half
nn.Conv2d(6, 16, 5)      # 6 input channels → 16 feature maps, 5×5 filters
nn.MaxPool2d(2)           # shrink by half
nn.Linear(256, 120)       # 256 flattened features → 120 neurons
nn.Linear(120, 84)        # 120 → 84 neurons
nn.Linear(84, 10)         # 84 → 10 digits (output)
```

## Historical note: activation functions

LeCun used **tanh**, not sigmoid, not ReLU. Tanh outputs between -1 and 1 (centered around zero), which trained better than sigmoid for this architecture. ReLU wouldn't be invented until 2010 and wouldn't become standard until AlexNet in 2012 — that's Act 3's story.

## The result

```
Test Accuracy: 0.9530
```

95.3% accuracy on handwritten digits it has never seen. A 1998 architecture, reading handwriting, getting 19 out of 20 correct.

The world mostly ignored this for 14 years. The computers weren't fast enough to go deeper. That changed when GPUs entered the picture.

## Key Insight

CNNs solve the spatial problem that regular neural networks can't — by using filters that see local neighborhoods, they understand that an image is made of patterns, not independent pixels. The network learns its own features instead of relying on a human to hand-craft them. This is the beginning of the end for feature engineering.