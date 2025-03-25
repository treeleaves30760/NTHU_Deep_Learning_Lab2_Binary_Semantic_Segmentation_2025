# Binary Semantic Segmentation Project

This project implements binary semantic segmentation models using various UNet architectures. It's designed to segment objects from backgrounds in images, specifically using the Oxford-IIIT Pet dataset as an example.

## Table of Contents

- [Binary Semantic Segmentation Project](#binary-semantic-segmentation-project)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
  - [Model Architectures](#model-architectures)
  - [Results](#results)

## Overview

Binary semantic segmentation is the task of classifying each pixel in an image as either belonging to a foreground object or the background. This project provides implementations of different UNet architectures to perform this task.

The models are trained and evaluated using the Dice coefficient, which measures the overlap between the predicted segmentation mask and the ground truth mask.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

Required dependencies:
- torch>=1.7.0
- torchvision>=0.8.0
- numpy>=1.19.0
- matplotlib>=3.3.0
- pillow>=8.0.0
- tqdm>=4.50.0
- scikit-learn>=0.23.0
- tensorboard

## Dataset

This project uses the Oxford-IIIT Pet dataset, which consists of images of pets (cats and dogs) with pixel-level segmentation masks.

The dataset will be automatically downloaded the first time you run the training script if it's not already present in the specified directory.

## Project Structure

```
├── dataset/                  # Dataset storage directory
│   └── oxford-iiit-pet/      # Oxford-IIIT Pet dataset
├── logs/                     # Training logs and learning curves
├── results/                  # Inference results and visualizations
├── saved_models/             # Saved model checkpoints
├── src/                      # Source code
│   ├── models/               # Model architectures
│   │   ├── unet.py           # UNet implementation
│   │   └── resnet34_unet.py  # ResNet34-UNet implementation
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── inference.py          # Inference script
│   ├── oxford_pet.py         # Dataset loading utilities
│   └── utils.py              # Utility functions
└── requirements.txt          # Required packages
```

## Usage

### Training

To train a model, use the `train.py` script:

```bash
python src/train.py --model_type unet --epochs 30 --batch_size 16 --learning-rate 3e-4 --data_path ./dataset/oxford-iiit-pet
```

Available command-line arguments:
- `--model_type`: Model architecture (`unet`, `unet_large`, `unet_xl`, or `resnet34_unet`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning-rate`: Learning rate
- `--data_path`: Path to the dataset

During training, the model with the best validation Dice score will be saved automatically. Training logs and learning curves are saved in the `logs` directory.

### Evaluation

The evaluation is performed automatically during training, but you can also use the evaluation functionality separately:

```python
from src.evaluate import evaluate
from src.models.unet import UNet
import torch

# Initialize model
model = UNet(n_channels=3, n_classes=1, bilinear=True)
model.load_state_dict(torch.load('path/to/model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate
dice_score = evaluate(model, data_loader, device)
print(f'Dice Score: {dice_score:.4f}')
```

### Inference

To run inference on test images, use the `inference.py` script:

```bash
python src/inference.py --model path/to/model.pth --model_type unet --num_samples 100 --output_dir results
```

Available command-line arguments:
- `--model`: Path to the saved model weights
- `--model_type`: Model architecture
- `--data_path`: Path to the dataset
- `--batch_size`: Batch size
- `--num_samples`: Number of test samples to process
- `--output_dir`: Directory to save the results
- `--save_results`: Whether to save the visualization results

Inference results, including the original image, ground truth mask, and predicted mask, will be saved in the `results` directory.

## Model Architectures

This project implements several variants of the UNet architecture:

1. **UNet**: The original UNet architecture with configurable channels.
2. **UNet Large**: A larger version of UNet with more filters.
3. **UNet XL**: An extra-large version of UNet for more complex segmentation tasks.
4. **ResNet34-UNet**: A UNet architecture with a ResNet34 encoder, which can be pre-trained on ImageNet.

All models can be configured with bilinear upsampling or transposed convolutions for the decoder.

## Results

The models are evaluated using the Dice coefficient, which measures the overlap between the predicted segmentation mask and the ground truth mask. Higher Dice scores indicate better segmentation performance.

During training, learning curves showing the training loss, validation Dice score, and learning rate are automatically generated and saved in the `logs` directory.

After inference, visualization results showing the original image, ground truth mask, and predicted mask are saved in the `results` directory.

---

This project is designed for educational and research purposes in the field of deep learning and computer vision. 