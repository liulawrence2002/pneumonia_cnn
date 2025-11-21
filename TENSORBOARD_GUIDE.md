# TensorBoard Visualization Guide

## What is TensorBoard?

TensorBoard is TensorFlow's visualization toolkit that allows you to monitor your model's training in real-time and analyze its architecture and performance.

## Quick Start

### 1. Train Your Model
```bash
python src/training_rgb_pos_neg.py
```

This will:
- Train your pneumonia detection model
- Save training logs to `logs/fit/` directory
- Generate a model architecture diagram as `model_architecture_posneg.png`

### 2. Launch TensorBoard

**Easiest Method:**
Double-click [launch_tensorboard.bat](launch_tensorboard.bat)

**Or run manually:**
```bash
# Activate virtual environment
venv\Scripts\activate

# Launch TensorBoard
tensorboard --logdir=logs/fit
```

### 3. Open Your Browser
Navigate to: **http://localhost:6006**

## What You Can Visualize

### SCALARS Tab
View metrics over time:
- **epoch_accuracy**: Training accuracy per epoch
- **epoch_loss**: Training loss per epoch
- **epoch_val_accuracy**: Validation accuracy per epoch
- **epoch_val_loss**: Validation loss per epoch
- **epoch_learning_rate**: Learning rate changes (from ReduceLROnPlateau callback)

**Use Cases:**
- Monitor if your model is learning (accuracy should increase, loss should decrease)
- Detect overfitting (training accuracy high, validation accuracy low)
- See when early stopping triggered
- Track learning rate reductions

### GRAPHS Tab
Interactive visualization of your neural network architecture:
- See all layers and their connections
- Click on layers to see their parameters
- Understand data flow through the network

### DISTRIBUTIONS Tab
Statistical distributions of weights and biases:
- See how weights evolve during training
- Identify if weights are updating properly
- Detect vanishing or exploding gradients

### HISTOGRAMS Tab
3D histograms showing weight distributions across training:
- View weight distributions over time
- See how layers change during training
- Identify dead neurons or saturation

## Comparing Multiple Training Runs

TensorBoard can compare different training sessions:

1. Each training run creates a timestamped folder in `logs/fit/`
2. TensorBoard loads all runs automatically
3. Toggle runs on/off in the left sidebar
4. Compare different hyperparameters or architectures

Example folder structure:
```
logs/
└── fit/
    ├── 20250119-143022/  # First training run
    ├── 20250119-150845/  # Second training run
    └── 20250119-162133/  # Third training run
```

## Understanding the Metrics

### Accuracy
- **Higher is better** (0.0 to 1.0, or 0% to 100%)
- Training accuracy: How well the model fits training data
- Validation accuracy: How well the model generalizes to unseen data

### Loss
- **Lower is better**
- Measures how wrong the model's predictions are
- Should decrease over time
- Validation loss increasing = overfitting

### Learning Rate
- Starts at initial value (1e-4 in your model)
- Reduces when validation accuracy plateaus
- Smaller learning rate = smaller updates to weights

## Tips for Using TensorBoard

### 1. Smoothing
Use the slider in the left panel to smooth noisy curves:
- Smoothing = 0: Raw data
- Smoothing = 0.6-0.8: Good balance
- Smoothing = 0.99: Very smooth, may hide details

### 2. Refresh
TensorBoard auto-refreshes every 30 seconds during training
- Click the refresh button for immediate update

### 3. Download Data
Click the download button (↓) to export charts as:
- SVG (vector graphics)
- CSV (raw data)
- JSON (structured data)

### 4. Axis Scaling
- Toggle between linear and log scale
- Useful for comparing runs with different ranges

## Model Architecture Visualization

The script also saves a static PNG diagram:
- File: `model_architecture_posneg.png`
- Shows all layers with their shapes
- Displays connections between layers
- Includes layer parameters

Open this file to get a quick overview of your model structure without running TensorBoard.

## Troubleshooting

### TensorBoard won't start
**Error:** "No dashboards are active"
- **Solution:** Train your model first to generate logs

### Port already in use
**Error:** "Address already in use"
- **Solution:** Use a different port:
  ```bash
  tensorboard --logdir=logs/fit --port=6007
  ```

### Can't see graphs
- Check that `logs/fit/` directory exists and has timestamped folders
- Ensure training ran long enough to generate at least one epoch
- Try refreshing the browser (Ctrl+F5)

### Model architecture diagram not generating
**Error:** "Failed to import pydot"
- **Solution:** Install graphviz:
  1. Install Graphviz from: https://graphviz.org/download/
  2. Add Graphviz to PATH
  3. Reinstall pydot: `pip install --upgrade pydot`

## Advanced Usage

### Custom Port
```bash
tensorboard --logdir=logs/fit --port=8080
```

### Bind to All Network Interfaces
```bash
tensorboard --logdir=logs/fit --host=0.0.0.0
```
Access from other devices: `http://your-ip:6006`

### Load Specific Run
```bash
tensorboard --logdir=logs/fit/20250119-143022
```

## Best Practices

1. **Train multiple times** with different hyperparameters and compare
2. **Check validation metrics** more than training metrics
3. **Look for overfitting**: Training accuracy >> Validation accuracy
4. **Monitor learning rate**: Should decrease when stuck
5. **Use smoothing** to see trends in noisy data
6. **Save screenshots** of good training runs for documentation

## Resources

- Official TensorBoard Guide: https://www.tensorflow.org/tensorboard
- TensorBoard Tutorial: https://www.tensorflow.org/tensorboard/get_started
