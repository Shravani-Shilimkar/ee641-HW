Name: Shravani Sunil Shilimkar
USC email address: shilimka@usc.edu


## HW1 Q1

# Multi-Scale Object Detection Implementation

This repository contains a complete implementation of a multi-scale single-shot object detector for synthetic shape detection.

## Overview

The system detects three types of shapes at different scales:
- **Circles** (small objects, 15-40px)
- **Squares** (medium objects, 35-70px)  
- **Triangles** (large objects, 60-100px)

## Architecture

### Multi-Scale Feature Extraction
- **Scale 1** (56×56): Detects small objects with anchor scales [16, 24, 32]
- **Scale 2** (28×28): Detects medium objects with anchor scales [48, 64, 96]
- **Scale 3** (14×14): Detects large objects with anchor scales [96, 128, 192]

### Model Components
1. **Backbone**: 4-block CNN with progressive downsampling
2. **Detection Heads**: Per-scale prediction heads for bbox regression and classification
3. **Anchor System**: Multi-scale anchors with 1:1 aspect ratio

## Setup and Usage

### 1. Generate Dataset
```bash
python generate_datasets.py --seed 641 --num_train 1000 --num_val 200
```

### 2. Train Model
```bash
python train.py
```

Training configuration:
- Batch size: 16
- Learning rate: 0.001 (SGD with momentum 0.9)
- Epochs: 50
- Loss weights: Objectness=1.0, Classification=1.0, Localization=2.0

### 3. Evaluate Model
```bash
python evaluate.py
```

## File Structure

```
problem1/
├── dataset.py              # COCO-style dataset loader
├── model.py               # Multi-scale detector architecture
├── utils.py               # Anchor generation and box utilities
├── loss.py                # Multi-task detection loss
├── train.py               # Training script with logging
├── evaluate.py            # Evaluation and visualization
├── generate_datasets.py   # Synthetic data generation
└── results/
    ├── best_model.pth        # Trained model weights
    ├── training_log.json     # Training metrics
    ├── evaluation_results.json # mAP scores
    └── visualizations/       # Detection visualizations
```

## Key Features

### Loss Function
- **Objectness Loss**: Binary cross-entropy with hard negative mining (3:1 ratio)
- **Classification Loss**: Cross-entropy for positive anchors only
- **Localization Loss**: Smooth L1 loss for positive anchors only

### Anchor Matching
- Positive anchors: IoU ≥ 0.5 with ground truth
- Negative anchors: IoU < 0.3 with all ground truth boxes
- Hard negative mining to balance positive/negative samples

### Post-Processing
- Confidence thresholding (0.5)
- Per-class Non-Maximum Suppression (IoU threshold 0.5)
- Bounding box clamping to image bounds

## Training Details

### Data Augmentation
- ImageNet normalization
- No geometric augmentation (to maintain precise shape properties)

### Optimization
- SGD optimizer with momentum 0.9
- Learning rate scheduling: StepLR (γ=0.1 at epoch 20)
- Weight decay: 1e-4

### Monitoring
- Training/validation loss curves saved to JSON
- Best model saved based on validation loss
- Detailed per-component loss tracking (objectness, classification, localization)

## Evaluation Metrics

### Average Precision (AP)
- Computed per class using 11-point interpolation
- IoU threshold: 0.5
- Mean AP across all three classes

### Visualizations
- Detection results on validation images
- Ground truth vs predictions comparison
- Bounding box visualization with confidence scores

## Implementation Notes

### Scale Specialization
The model is designed so different scales naturally specialize for different object sizes:
- Scale 1 (56×56) with small anchors → detects circles
- Scale 2 (28×28) with medium anchors → detects squares  
- Scale 3 (14×14) with large anchors → detects triangles

### Memory Efficiency
- Batch processing with proper memory management
- Gradient accumulation for larger effective batch sizes if needed
- Device-agnostic implementation (CPU/GPU)

## Expected Results

With the synthetic dataset, you should achieve:
- **Circle AP**: ~0.85-0.95
- **Square AP**: ~0.85-0.95  
- **Triangle AP**: ~0.85-0.95
- **Mean AP**: ~0.90+

The high performance is expected due to the controlled synthetic nature of the data and clear scale separation between object types.

## Extensions

Potential improvements:
1. **Multi-aspect ratios**: Add 1:2, 2:1 aspect ratios
2. **Feature Pyramid Networks**: Add top-down connections
3. **Focal Loss**: Replace hard negative mining
4. **Data augmentation**: Add rotation, scaling, color jittering
5. **Anchor-free detection**: Implement FCOS-style detection

## Dependencies

```
torch >= 1.9.0
torchvision >= 0.10.0
PIL >= 8.0.0
matplotlib >= 3.3.0
numpy >= 1.19.0
tqdm >= 4.60.0
```

## Citation

This implementation is based on standard single-shot detection principles with multi-scale feature pyramids, similar to SSD and RetinaNet architectures.





## HW1 Q2

# Problem 2: Heatmap vs. Direct Regression for Keypoint Detection
This project implements and compares two deep learning approaches for 2D keypoint detection on a synthetic "stick figure" dataset:

Spatial Heatmap Regression: The model predicts a 2D Gaussian heatmap for each keypoint, where the location of the peak corresponds to the keypoint's coordinates.

Direct Coordinate Regression: The model directly regresses the normalized (x, y) coordinates for all keypoints.

The goal is to quantify the performance difference and analyze the strengths and weaknesses of each method.

## File Structure
problem2/
├── data/                     # Generated synthetic data
├── results/                  # Saved models, logs, and visualizations
│   ├── training_log.json
│   ├── heatmap_model.pth
│   ├── regression_model.pth
│   └── visualizations/
├── dataset.py                # PyTorch Dataset for loading data
├── model.py                  # HeatmapNet and RegressionNet model definitions
├── train.py                  # Script to train both models
├── evaluate.py               # Script to evaluate models and generate plots
├── baseline.py               # Script for ablation studies and failure analysis
├── generate_data.py          # Script to create the synthetic dataset
└── README.md                 # This file
## Setup and Installation
### 1. Prerequisites
Python 3.8+

Git

### 2. Clone Repository
Clone your project repository to your local machine.

Bash

git clone https://github.com/your-username/ee641-hw1-your-username.git
cd ee641-hw1-your-username/problem2/
### 3. Create Virtual Environment
It is highly recommended to use a virtual environment.

Bash

# Create the environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
.\venv\Scripts\activate
### 4. Install Dependencies
Install the required libraries using the provided requirements.txt file.

# Create a requirements.txt file with the content below
# torch, torchvision, numpy, matplotlib, Pillow, tqdm
pip install -r requirements.txt
For GPU support, please install the appropriate version of PyTorch by following the instructions on the official PyTorch website.

## Usage Guide
### Step 1: Generate the Dataset 🤖
First, run the data generation script to create the synthetic training and testing images.

Bash

python generate_data.py
This command will create a data/ directory containing train and test subdirectories, each with an images folder and an annotations.json file.

### Step 2: Train the Models ⏳
Next, train both the heatmap and regression models. This script will save the best model checkpoints and a log of the training progress.

Bash

python train.py
Models (heatmap_model.pth, regression_model.pth) and logs (training_log.json) will be saved in the results/ directory.

### Step 3: Evaluate and Visualize Results 📈
After training, run the evaluation script to compare the models' performance using the PCK (Percentage of Correct Keypoints) metric and to generate visualizations.

Bash

python evaluate.py
This will generate the PCK curve, sample prediction images, and save them in results/visualizations/.

### Step 4: Further Analysis (Optional) 🔬
To perform ablation studies or analyze specific failure cases, use the baseline.py script.

Bash

python baseline.py
This saves failure case visualizations to results/visualizations/failures/.

## Results and Analysis
The project evaluates the models by comparing their PCK curves and visualizing their predictions.

### PCK Performance
The heatmap-based approach consistently outperforms direct regression across all thresholds. This suggests that preserving spatial information through heatmaps provides a more robust representation for the network to learn from.

### Heatmap Learning Progression
The heatmap model learns to produce sharp, confident peaks centered on the keypoints as training progresses.

[Image showing heatmap evolution over epochs]

### Qualitative Comparison
Visual inspection shows the heatmap model's predictions are generally more accurate and less prone to catastrophic failures than the direct regression model.

### Failure Case Analysis
The direct regression model is more likely to fail by predicting an "average" pose or collapsing keypoints to a single location, especially when the input is ambiguous. The heatmap model, while more accurate, can sometimes struggle with closely-spaced keypoints, producing a merged or indistinct heatmap peak.