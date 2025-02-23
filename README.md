# Fashion Similarity Model & Lookalike Phishing Detection

## Overview
This repository contains two projects:

1. **Fashion Similarity Model** - A deep learning model for comparing the similarity between two pieces of clothing using the Fashion MNIST dataset.
2. **Lookalike Phishing Detection** - A convolutional neural network (CNN) model designed to detect phishing websites that visually resemble legitimate banking and payment portals.

## Table of Contents
- [Fashion Similarity Model](#fashion-similarity-model)
  - [Objective](#objective)
  - [Implementation](#implementation)
  - [Usage](#usage)
- [Lookalike Phishing Detection](#lookalike-phishing-detection)
  - [Objective](#objective-1)
  - [Implementation](#implementation-1)
  - [Usage](#usage-1)
- [Requirements](#requirements)
- [Installation](#installation)
- [License](#license)

## Fashion Similarity Model
### Objective
The Fashion Similarity Model aims to compare the similarity between different clothing items using a convolutional neural network (CNN) and cosine similarity. This model helps in applications such as recommending similar fashion items in e-commerce.

### Implementation
- The **Fashion MNIST dataset** is used as the base dataset.
- A **CNN model** is trained to extract feature embeddings from clothing images.
- The **cosine similarity metric** is applied to measure the closeness between two images based on their feature vectors.

### Usage
1. Load the dataset and preprocess the images.
2. Train the CNN model to extract image features.
3. Compute the cosine similarity between pairs of clothing images.
4. Use the model to find similar fashion items.

## Lookalike Phishing Detection
### Objective
The Lookalike Phishing Detection model is designed to detect phishing websites that visually resemble legitimate banking and payment websites. It focuses on image-based detection using CNNs.

### Implementation
- A **CNN-based classifier** is trained to distinguish between phishing and legitimate websites.
- The model extracts features from website screenshots and uses them for classification.
- The **cosine similarity metric** is also used to compare website images against a dataset of known legitimate sites.

### Usage
1. Collect website images (phishing vs. legitimate).
2. Train the CNN model to classify phishing websites.
3. Use the trained model to compare a suspicious site’s similarity to known legitimate sites.
4. Detect potential phishing threats based on similarity scores.

## Requirements
- Python 3.8+
- PyTorch
- TensorFlow (optional for extended use cases)
- NumPy
- Matplotlib
- Scikit-learn
- torchvision
- OpenCV (for image processing)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fashion-phishing-detection.git
   cd fashion-phishing-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the models:
   ```bash
   python fashion_similarity.py  # For Fashion Similarity Model
   python phishing_detection.py  # For Lookalike Phishing Detection
   ```


