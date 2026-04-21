# Prototypical Networks for Few-Shot Image Classification

A from-scratch PyTorch implementation of Prototypical Networks for few-shot classification, trained and evaluated on CIFAR-10.

Based on the paper: [Prototypical Networks for Few-Shot Learning](https://arxiv.org/abs/1703.05175) (Snell et al., 2017).

## Results

- **62% accuracy on 5-way 5-shot classification** (random baseline: 20%)
- Trained for 3000 episodes on CIFAR-10 training set
- Evaluated on CIFAR-10 test set

## How It Works

Prototypical Networks learn an embedding space where images of the same class cluster together. At inference time, a small **support set** of labeled examples is used to compute a **prototype** (mean embedding) for each class. A query image is then classified by finding the nearest prototype.

This makes the model useful for **few-shot learning** — classifying images from classes seen only a handful of times, without retraining.

## Architecture

A small CNN embedding network:

```
Conv2d(3, 64, 3x3) → BatchNorm2d → MaxPool2d(2x2)
Conv2d(64, 64, 3x3) → BatchNorm2d → MaxPool2d(2x2)
Conv2d(64, 64, 3x3) → BatchNorm2d → MaxPool2d(2x2)
Flatten
Linear(1024, 64) → ReLU
Linear(64, 64)
```

Output: 64-dimensional embedding vector per image.

## Training

Episodes are sampled following the episodic training procedure from the paper:

- **5-way 5-shot**: 5 classes per episode, 5 support examples per class
- **3 query images** per class per episode
- Support and query sets are guaranteed to be disjoint
- Loss: negative log-probability of the correct class under a softmax over prototype distances

**Optimizer:** Adam (lr=0.001)  
**Scheduler:** StepLR — lr halved every 1000 episodes

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 50,000 training images and 10,000 test images across 10 classes (32x32 RGB).

Download the Python version and place the `cifar-10-batches-py` folder in the project directory.

### Training

Open and run `prototypes.ipynb`. The trained model is saved as `proto.pt`.

### Validation

Open and run `validation.ipynb`. Loads `proto.pt` and evaluates over 2000 sampled episodes on the CIFAR-10 test set.

