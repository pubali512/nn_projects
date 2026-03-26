# Object Detection — Person Recognition (Faster R-CNN)

## Project Overview

A parameterized **person detection system** using **Faster R-CNN with ResNet-50 FPN** pretrained on COCO, fine-tuned for person detection. The system trains on a configurable subset of the COCO 2017 dataset, evaluates detection quality using mean IoU (Intersection over Union), and provides both 2D and 3D visualizations of training metrics.

### Main Features

- **Fine-tuned Faster R-CNN** — Replaces the box predictor head for 2-class detection (background + person) using COCO_V1 pretrained weights
- **Configurable training** — Parameterizable training/validation image ranges, epochs, batch size, and learning rate
- **COCO dataset integration** — Custom `CocoPedestrian` dataset class that wraps `torchvision.datasets.CocoDetection` and filters for person class only
- **Data augmentation** — Random horizontal flip during training
- **Evaluation metrics** — Mean IoU between predicted and ground truth bounding boxes
- **Visualization** — Draws bounding boxes with confidence scores on validation images
- **3D metric plots** — Scatter plots of epoch × num_training_images × accuracy (IoU) and epoch × num_images × loss
- **CUDA support** — Automatically uses GPU if available

## Technologies Used & Installation Instructions

### Technologies

- **Python 3**
- **PyTorch** (`torch`, `torchvision`) — Deep learning framework
- **Faster R-CNN** (ResNet-50 FPN, COCO_V1 weights) — Object detection model
- **torchvision.ops** — Box IoU and NMS operations
- **matplotlib** — 2D and 3D visualization
- **NumPy** — Numerical computing
- **COCO 2017 Dataset** — Training and validation data (must be downloaded separately)

### Prerequisites

- Python 3.x
- pip or conda
- **COCO 2017 dataset** downloaded locally (see below)

### Installation

```bash
cd nn_projects/object_detect

# Install dependencies
pip install torch torchvision matplotlib numpy
```

### COCO 2017 Dataset Setup

Download the COCO 2017 dataset from [cocodataset.org](https://cocodataset.org/#download):

1. `train2017.zip` — Training images
2. `val2017.zip` — Validation images
3. `annotations_trainval2017.zip` — Annotation files

Extract them and note the paths to:
- `train2017/` (training images directory)
- `val2017/` (validation images directory)
- `instances_train2017.json` (training annotations)
- `instances_val2017.json` (validation annotations)

## Usage Instructions

### Configuration

Before running, update the dataset paths in the `__main__` block of `personRecogParamMain.py`:

```python
detector = PersonDetector(
    train_root="path/to/train2017",
    train_ann="path/to/annotations/instances_train2017.json",
    val_root="path/to/val2017",
    val_ann="path/to/annotations/instances_val2017.json",
    train_start_idx=2000,
    train_end_idx=2100,
    val_start_idx=500,
    val_end_idx=600,
    num_epochs=6,
    batch_size=2,
    learning_rate=0.005
)
```

### Running

```bash
python personRecogParamMain.py
```

The script will:
1. Load and filter COCO data for person-class annotations
2. Fine-tune Faster R-CNN on the training subset
3. Evaluate on the validation subset using mean IoU
4. Display bounding box visualizations on validation images
5. Generate 3D plots of training metrics

### Default Configuration

| Parameter            | Default Value |
|----------------------|---------------|
| Training images      | 100 (indices 2000–2100) |
| Validation images    | 100 (indices 500–600)   |
| Epochs               | 6             |
| Batch size           | 2             |
| Learning rate        | 0.005         |

### Files

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `personRecogParamMain.py` | Main detection pipeline (dataset, model, training, evaluation, visualization) |
| `icons.py`                | Console output formatting utility (emoji + color codes) |
