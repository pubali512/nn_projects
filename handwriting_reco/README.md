# Handwriting Recognition (MNIST CNN)

## Project Overview

A Jupyter notebook implementing **handwritten digit classification** on the **MNIST dataset** using a parameterizable **Convolutional Neural Network (CNN)** built with PyTorch. The notebook covers the full pipeline from data loading and exploration through model training, evaluation, and hyperparameter benchmarking.

### Main Features

- **Parameterizable CNN** — Configurable convolutional layers (output channels), max pooling kernel size, dropout, and fully connected layer sizes with dynamic feature size computation
- **Flexible training** — Train for a fixed number of epochs or until a target accuracy is reached (with a maximum of 20 iterations)
- **Model checkpointing** — Saves model and optimizer state after training for later use
- **Prediction visualization** — Predict and display individual test images with their classified labels
- **Hyperparameter benchmarking** — Automated experiments with visualizations:
  - FC1 output features (25–75) vs. accuracy (2D plot)
  - Conv1 × Conv2 output channels vs. accuracy (3D scatter plot)
  - Conv2 output channels (40–100) vs. accuracy (2D plot)
  - Conv1 output channels (10–60) vs. accuracy (2D plot)

## Technologies Used & Installation Instructions

### Technologies

- **Python 3**
- **PyTorch** (`torch`, `torchvision`) — Neural network framework
- **torch.nn** / **torch.nn.functional** — Network layers and operations
- **torch.optim** (SGD) — Optimizer
- **matplotlib** — Visualization and plotting
- **Jupyter Notebook** — Interactive development environment

### Prerequisites

- Python 3.x
- pip or conda

### Installation

```bash
cd nn_projects/handwriting_reco

# Install dependencies
pip install torch torchvision matplotlib jupyter
```

The MNIST dataset is automatically downloaded on first run to a `./files/` directory.

## Usage Instructions

### Running the Notebook

Open the notebook in Jupyter or VS Code and run the cells sequentially:

```bash
jupyter notebook handwriting_notebook.ipynb
```

Or directly in VS Code with the Jupyter extension.

### Notebook Structure

1. **Data Loading** — Loads MNIST train (batch size 64) and test (batch size 1000) sets with normalization
2. **Data Exploration** — Functions to retrieve and display images by batch/image index, display random samples
3. **Model Definition (`ParamCNN`)** — A configurable CNN with:
   - 2 convolutional layers (configurable output channels)
   - Max pooling (configurable kernel size)
   - Dropout layer
   - 2 fully connected layers (configurable FC1 size, FC2 = 10 for digit classes)
4. **Training (`MnistClassifier`)** — Wrapper class that trains:
   - For a fixed number of epochs, or
   - Until a target accuracy is reached (max 20 iterations)
5. **Prediction** — Classify individual test images and display results
6. **Benchmarking** — Sweep hyperparameters and plot accuracy trends (2D and 3D visualizations)

### Output

- Model checkpoint: `./results/model.pth`
- Optimizer checkpoint: `./results/optimizer.pth`
- MNIST data: auto-downloaded to `./files/`
