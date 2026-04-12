# The Architecture Wars (2014–2015)

## Deeper = better... right?

After AlexNet, the logic seemed obvious. More layers = more abstract features = better accuracy. Everyone started stacking layers.

## The race for depth

- **VGGNet (2014)** — Oxford's approach: radical simplicity. Just use 3×3 filters everywhere. AlexNet used 11×11 and 5×5, but VGG showed that stacking two 3×3 layers sees the same area as one 5×5 — fewer parameters, more non-linearity (extra ReLU in between). Went to 16 and 19 layers. 7.3% error on ImageNet. The downside: 138 million parameters. Slow and memory-hungry.

- **GoogLeNet / Inception (2014)** — Google's approach: why choose one filter size? Run 1×1, 3×3, and 5×5 filters **in parallel** at each layer and concatenate the results. Let the network decide which scale matters. 22 layers, 6.7% error — with only 5 million parameters. Way more efficient than VGG. But the architecture was complex and hard to modify.

```
2012  AlexNet      8 layers    15.3% error
2014  VGGNet      19 layers     7.3% error   ← simpler filters, more depth
2014  GoogLeNet   22 layers     6.7% error   ← parallel filters, efficient
```

## It breaks at 56 layers

Then people tried 56 layers. Something bizarre happened — it performed **worse** than a 20-layer network. Not just on test data — on **training data** too.

This wasn't overfitting. If it were, training accuracy would be high and test accuracy low. But both were bad. The deeper network couldn't even learn what the shallower one already knew.

A 56-layer network *should* be at least as good as a 20-layer one. In theory, the extra 36 layers could just pass data through unchanged — identity functions. But the network couldn't even learn that. The optimization landscape was too complex, and gradients weakened over that many layers — even with ReLU.

## ResNet — 152 layers. How!?

**Kaiming He** (2015, Microsoft Research) had an insight that was almost embarrassingly simple.

Instead of asking each layer to learn the full output, ask it to learn the **difference** — the residual. And carry the original input forward alongside it:

```
output = F(input) + input
```

That `+ input` is the **skip connection** (also called **residual connection** — that's where **Res**Net gets its name).

When a layer gets an input, its output has both the input and whatever the layer computed. If the layer's computation is useful — great, the output is improved. If not — no harm done, the input is still there. Because of ReLU, the worst a layer can output is 0. So `output = 0 + input = input`. The layer just steps aside.

The network only gets better by adding layers, never worse. We only keep the good results.

The gradient benefits too. During backpropagation:

```
gradient through (F(input) + input) = gradient through F(input) + 1
```

That `+ 1` means the gradient always has a clean highway backward, no matter how many layers it passes through. No vanishing. Every layer learns.

Result: **152 layers. 3.57% error.** Better than average human performance.

```
2012  AlexNet      8 layers    15.3% error
2014  VGGNet      19 layers     7.3% error
2014  GoogLeNet   22 layers     6.7% error
2015  ResNet     152 layers     3.57% error  ← skip connections
```

## Batch normalization

ResNet uses **batch normalization** inside every block. The problem it solves: as weights update during training, each layer's outputs shift in range. The next layer is chasing a moving target — it learned to work with inputs around 2.0–5.0, but suddenly they're 10.0–15.0.

At 8 layers (AlexNet), this is manageable. At 152 layers, the signals would drift so far that even skip connections couldn't save you.

Batch norm fixes this by normalizing each layer's output:

```
input → Conv → BatchNorm → ReLU → Conv → BatchNorm → + input → ReLU

1. Calculate the mean of all values in this batch
2. Calculate the standard deviation
3. Subtract the mean, divide by the standard deviation
4. Scale and shift by two learnable parameters (gamma and beta)
```

```
nn.BatchNorm2d(96) # normalize 96 feature maps
```

Step 4 is key — `gamma` and `beta` are learned through backpropagation. The network can adjust the normalization if it wants to. Batch norm doesn't force a specific range — it provides a stable starting point.

## The residual block

ResNet doesn't think in individual layers — it organizes them into repeating **residual blocks**:

```
input
  ├──→ Conv → BatchNorm → ReLU → Conv → BatchNorm ──→ ADD → ReLU → output
  │                                                     ↑
  └───────────────── skip connection ───────────────────┘
```

Two conv layers, two batch norms, a ReLU, and a skip connection — that's one block. A self-contained unit that can only help or do nothing. Never hurt.

- **ResNet-18**: 8 residual blocks (2 conv layers each = ~18 layers total)
- **ResNet-152**: 50 residual blocks (3 conv layers each = ~152 layers total)

Same design, stacked deeper. Like Lego bricks — same shape, piled higher.

## Transfer learning

The other huge innovation of this era. Training ResNet from scratch takes days on GPUs. But someone already did that work — why repeat it?

**Transfer learning**: take a model pretrained on millions of images, swap the last layer, and fine-tune for your task. The early layers already know edges, textures, shapes — universal visual features useful for *any* image task. Only the final layers need to be task-specific.

It's like hiring someone who already knows how to see, and just teaching them which 10 things you care about.

**PyTorch has pretrained models ready to go:**

```python
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # pretrained on ImageNet
model.fc = nn.Linear(model.fc.in_features, 10)                    # swap last layer for CIFAR-10
```

Two lines. A full ResNet-18 with weights trained on 1.2 million images, adapted to your task. Training time: minutes instead of days. Accuracy: better than training from scratch, because the base model learned from data you don't have.

Before transfer learning, every new task meant training from scratch. After — you download, swap, fine-tune. This changed everything.

## Key Insight

The architecture wars proved that going deeper works — but only with the right infrastructure. Skip connections let layers gracefully step aside. Batch normalization keeps signals stable. Transfer learning means you don't have to start from zero. Depth went from 8 layers to 152, and error dropped below human performance.