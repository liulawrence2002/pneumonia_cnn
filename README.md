# Pneumonia Classification CNN

A deep learning project for detecting and classifying pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs).

## Project Overview

This project implements multiple CNN architectures to classify chest X-ray images into different pneumonia categories:
- Normal (no pneumonia)
- Bacterial pneumonia
- Viral pneumonia

The models achieve over 95% accuracy in detecting the presence or absence of pneumonia.

## Project Structure

```
pneumonia_cnn/
├── src/
│   ├── training_RGBv2.py      # RGB-based CNN training with residual connections
│   ├── training_greyscale.py   # Greyscale-based CNN training
│   └── testing_rgbv2.py        # Model evaluation script
├── Data/
│   ├── PatientsTrainingData.xlsx  # Training dataset metadata
│   └── PatientsTestingData.xlsx   # Testing dataset metadata
├── images/                     # X-ray image files
├── venv/                       # Virtual environment
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Features

### Model Architectures

1. **RGB Model (v2)** - `training_RGBv2.py`
   - Input: 224x224x3 RGB images
   - Residual connections for improved gradient flow
   - Parallel convolutional paths in final layers
   - Data augmentation (rotation, shift, zoom, flip)
   - Regularization: Dropout, L2, Batch Normalization

2. **Greyscale Model** - `training_greyscale.py`
   - Input: 448x448x1 greyscale images
   - Adaptive image padding to preserve aspect ratio
   - Multiple convolutional blocks with BatchNorm
   - Binary classification for pneumonia presence/absence

### Key Features

- Data augmentation for improved generalization
- Early stopping and learning rate reduction callbacks
- Model checkpointing to save best performing models
- Comprehensive evaluation with classification reports
- Support for both 2-class and 3-class classification

## Installation

### 1. Clone or navigate to the project directory

```bash
cd pneumonia_cnn
```

### 2. Create and activate virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

**RGB Model (Recommended):**
```bash
python src/training_RGBv2.py
```

**Greyscale Model:**
```bash
python src/training_greyscale.py
```

Training will:
- Load and preprocess X-ray images from the `images/` directory
- Split data into training/validation/test sets (64%/16%/20%)
- Apply data augmentation to training data
- Train the CNN model with callbacks
- Save the best model as `best_pneumonia_modelrgbv2.keras` or `best_pneumonia_model.keras`
- Display test accuracy and classification report

### Testing a Model

```bash
python src/testing_rgbv2.py
```

This will:
- Load the trained model
- Preprocess test images
- Make predictions on test set
- Display accuracy and per-image predictions

## Model Performance

The RGB v2 model achieves approximately 95% accuracy in detecting pneumonia presence/absence on the test set.

## Data Requirements

The project expects:
- Excel files (`PatientsTrainingData.xlsx`, `PatientsTestingData.xlsx`) in the `Data/` folder
- X-ray images in the `images/` folder
- Excel columns: `Patient X-Ray File`, `Pneumonia`
- Pneumonia labels: 0 (normal), 1 (bacterial), 2 (viral)

## Dependencies

- TensorFlow 2.13+
- NumPy
- Pandas
- OpenCV (cv2)
- scikit-learn
- Pillow (PIL)
- imageio
- openpyxl

See [requirements.txt](requirements.txt) for specific versions.

## Training Configuration

### RGB Model v2
- Image size: 224x224x3
- Batch size: 32
- Optimizer: Adam (lr=1e-4, clipnorm=1.0)
- Loss: Sparse categorical crossentropy
- Epochs: 150 (with early stopping)
- Data augmentation: rotation (30°), shift (15%), zoom (20%), horizontal flip

### Greyscale Model
- Image size: 448x448x1
- Batch size: 16
- Optimizer: Adam (lr=1e-4)
- Loss: Sparse categorical crossentropy
- Epochs: 75 (with early stopping)
- Data augmentation: rotation (10°), shift (10%), zoom (10%), horizontal flip

## Model Outputs

Trained models are saved as:
- `best_pneumonia_modelrgbv2.keras` (RGB model)
- `best_pneumonia_model.keras` (Greyscale model)

These can be loaded using:
```python
from tensorflow.keras.models import load_model
model = load_model('best_pneumonia_modelrgbv2.keras')
```

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests for improvements.
