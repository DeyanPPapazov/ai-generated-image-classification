# AI-Generated Image Classification

A comprehensive study comparing different CNN architectures for distinguishing AI-generated images from real photographs, while avoiding reliance on trivial cues such as resolution, compression, or metadata artifacts.

## Table of Contents

- [Overview](#overview)
- [Goal](#goal)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)

## Overview

This project implements and compares three different approaches to classifying AI-generated vs. real images:

1. **CNN from Scratch** - A manually implemented convolutional neural network with explicit forward and backward passes
2. **Frozen ResNet-18** - A pretrained ResNet-18 model with frozen layers except the last trainable classification head
3. **Fine-tuned ResNet-18** - A pretrained ResNet-18 with the final residual block (layer4) and classification head trainable

## Goal
Develop a robust classifier that learns genuine visual differences between AI-generated and real images, rather than exploiting dataset-specific shortcuts like image size or compression artifacts.

## Dataset

- **AI-Generated Images**: 539 images from various AI art generators
- **Real Images**: 435 real photographs
- **File formats**: JPG, JPEG, PNG, WEBP
- **Resolution**: Variable (200x300 to 8500×7000 pixels)

### Dataset Split

- **Training**: 70% of total samples
- **Validation**: 15% of total samples  
- **Test**: 15% of total samples

All splits are stratified to maintain class balance.

##  Methodology

### Data Preprocessing

To prevent the models from learning trivial shortcuts based on image dimensions or file properties, all images go through standardized preprocessing:

#### For Scratch CNN:
- Convert to grayscale (single channel)
- Resize to 32×32 pixels using bilinear interpolation
- Normalize pixel values to [0, 1]

#### For ResNet Models:
- Convert to RGB
- Resize to 224×224 pixels using bilinear interpolation
- Normalize using ImageNet mean and standard deviation

### Training Strategy

**Scratch CNN:**
- 8 or 16 convolutional filters
- Binary cross-entropy loss
- Learning rate: 0.01
- 20 epochs

**Frozen ResNet-18:**
- Pretrained on ImageNet
- Only final FC layer trainable
- Cross-entropy loss
- Adam optimizer (lr=0.001)
- 3 epochs

**Fine-tuned ResNet-18:**
- Pretrained on ImageNet
- Layer4 + FC layer trainable
- Cross-entropy loss
- Adam optimizer (lr=0.0001)
- 8 epochs

## Models

### 1. CNN from Scratch

A minimal CNN architecture implemented manually to understand learning dynamics:

```
Input (32×32 grayscale)
    ↓
Conv3x3 (8 filters)
    ↓
ReLU
    ↓
MaxPool2x2
    ↓
Fully Connected (1800 → 1)
    ↓
Sigmoid
    ↓
Class Output [Real | AI]
```

**Parameters**: ~14,500 trainable parameters

### 2. Frozen ResNet-18

```
Input (224×224 RGB)
    ↓
ResNet-18 Backbone (frozen)
    ↓
Fully Connected (512 → 2)
    ↓
Softmax
    ↓
Class Output [Real | AI]
```

**Parameters**: ~1,000 trainable parameters (FC layer only)

### 3. Fine-tuned ResNet-18

```
Input (224×224 RGB)
    ↓
ResNet-18 Layers 1-3 (frozen)
    ↓
ResNet-18 Layer 4 (trainable)
    ↓
Fully Connected (512 → 2)
    ↓
Softmax
    ↓
Class Output [Real | AI]
```

**Parameters**: ~3M trainable parameters (layer4 + FC)

## Results

### Accuracy Comparison

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| **Scratch CNN** | ~70% | - | ~63% |
| **Frozen ResNet-18** | ~73% | ~68% | ~69% |
| **Fine-tuned ResNet-18** | ~100% | ~82% | **~81%** |

### Confusion Matrix Analysis

**Scratch CNN:**
- High false positives and false negatives
- Limited discriminative power due to shallow architecture
- Grayscale input loses color-based cues

**Frozen ResNet-18:**
- Reduced false negatives (better AI detection)
- High false positives remain
- Generic ImageNet features provide baseline performance

**Fine-tuned ResNet-18:**
- **Best balanced error distribution**
- Dramatically reduced false positives
- Task-specific feature adaptation improves discrimination

### Training Dynamics

- **Scratch CNN**: Shows clear overfitting with training-test accuracy gap widening after epoch 5
- **Frozen ResNet-18**: Stable generalization with aligned train/val curves
- **Fine-tuned ResNet-18**: Some overfitting but maintains strong test performance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-generated-image-classification.git
cd ai-generated-image-classification

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

***Step 1:*** Download the dataset from Kaggle using [this link](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-imagesm)

***Step 2:*** In your project’s main directory, create a new folder called /data.

***Step 3***: Place the downloaded image folders inside the /data directory so the scripts can find them.

## Usage

### Training Individual Models

The notebook is organized into sections:

1. **Data Exploration and Familiarization** - Analyze image properties and distributions
2. **Scratch CNN** - Train custom CNN from scratch
3. **ResNet-18 (Frozen)** - Feature extraction baseline
4. **ResNet-18 (Fine-tuned)** - Task-specific fine-tuning
5. **Model Comparison** - Confusion matrices and analysis

## Project Structure

```
ai-generated-image-classification/
├── data/
│   ├── AiArtData/
│   │   └── AiArtData/          # AI-generated images
│   └── RealArt/
│       └── RealArt/            # Real photographs
├── notebook/
│   └── main_pipeline.ipynb     # Main analysis notebook
├── README.md
├── LICENSE
└── requirements.txt
```

##  Key Findings

### 1. Preprocessing Matters
Controlled preprocessing that standardizes image resolution is critical to prevent models from exploiting trivial shortcuts. Analysis showed:
- AI images: Mean 1200×1000 pixels
- Real images: Mean 1500×1300 pixels with heavy-tailed distribution

Without resizing, models could achieve high accuracy by simply learning size patterns.

### 2. Transfer Learning Advantages
- Generic ImageNet features (frozen ResNet-18) provide surprisingly good baseline (~69% accuracy)
- Fine-tuning task-specific layers yields +12% accuracy improvement
- Scratch CNN with limited data and compute struggles to compete

### 3. Model Capacity vs. Generalization
- Scratch CNN overfits despite small architecture
- Frozen features prevent overfitting but peak early
- Fine-tuned model balances capacity and generalization best

### 4. Learning Rate Sensitivity
Experiments with different learning rates for scratch CNN:
- **lr=0.01**: Steady convergence, ~63% test accuracy
- **lr=0.1**: Unstable training, ~55% test accuracy (baseline)
- Higher learning rates prevent effective gradient descent

### 5. Class-Specific Performance
Fine-tuned ResNet-18 shows:
- **High precision** for real images (low false positives)
- **High recall** for AI images (low false negatives)
- Balanced performance critical for production deployment

**Note**: *The manually implemented CNN serves primarily as an educational tool to understand convolution operations and backpropagation. For production use, the fine-tuned ResNet-18 is recommended.*