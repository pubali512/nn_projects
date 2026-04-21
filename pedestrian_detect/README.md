# Pedestrian Detect

> Python 3.10+ / tkinter / OpenCV / scikit-learn

---

## 1. Project Overview

This application trains a **Gaussian Naive Bayes classifier** to detect pedestrians in images using the [Daimler Mono Pedestrian Detection Benchmark](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html). It provides a tkinter GUI with two main tabs — **Train** and **Predict** — plus a log/status panel.

| Feature Type | Description |
|---|---|
| **Greyscale** | Raw pixel intensities of 18×36 images → 648-dim vector |
| **HOG** | Histogram of Oriented Gradients (configurable block/cell size, default 12×12 blocks, 6×6 cells, 9 bins) |

### Workflow
1. **Train** — Read pedestrian (18×36 PGM) and non-pedestrian images, extract features, train a Gaussian Naive Bayes classifier.
2. **Predict** — Slide a multi-scale window across test images, classify each patch, unify overlapping detections, compare with a reference (default HOG detector or file).
3. **Evaluate** — Generate confusion matrices, per-file details, and summary reports.

---

## 2. Getting Started

### Prerequisites

- **Python 3.10+** installed and available as `python3` (macOS/Linux) or `python` (Windows)

### First-Time Setup (any OS)

```bash
# Clone the repo and cd into it
git clone <repo-url>
cd pedestrian_detect

# Run the cross-platform setup script — creates .venv/ and installs all dependencies
python3 setup_env.py        # macOS / Linux
python  setup_env.py        # Windows
```

This will:
1. Create a virtual environment in `.venv/`
2. Upgrade `pip`
3. Install all packages from `src/requirements.txt` (OpenCV, NumPy, scikit-learn, Pillow)
4. Verify the installation

### Running the App

```bash
# Option A — use the launcher (recommended)
python3 run.py              # macOS / Linux
python  run.py              # Windows

# Option B — setup + launch in one step
python3 run.py --setup

# Option C — manual activation
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
cd src && python main.py
```

### Recreating the Environment

```bash
python3 setup_env.py --clean   # Deletes .venv/ and rebuilds from scratch
```

---

## 3. Directory Structure

```
pedestrian_detect/
│
├── README.md                      # This file
├── setup_env.py                   # Cross-platform setup script (creates .venv, installs deps)
├── run.py                         # Cross-platform app launcher
├── .gitignore
│
├── src/
│   ├── main.py                    # Entry point — launches the GUI
│   ├── requirements.txt           # Python dependencies
│   ├── demo_scene_detection.py    # Standalone demo: composite scene generation
│   ├── test_workflow.py           # Headless end-to-end test script
│   │
│   ├── core/                      # Domain logic
│   │   ├── config.py              # TrainOptions, PredictOptions, data classes, constants
│   │   ├── feature_extraction.py  # Greyscale & HOG feature vector extraction
│   │   ├── trainer.py             # Training pipeline (read → extract → fit GaussianNB)
│   │   ├── predictor.py           # Prediction pipeline (sliding window → classify → unify)
│   │   └── evaluator.py           # Confusion matrix, accuracy, report generation
│   │
│   ├── gui/                       # tkinter user interface
│   │   ├── app.py                 # Main application window (tk.Tk subclass)
│   │   ├── train_tab.py           # "Train" tab — controls and event handlers
│   │   ├── predict_tab.py         # "Predict" tab — controls and event handlers
│   │   ├── output_panel.py        # Scrollable log panel + status bar with progress
│   │   └── result_dialogs.py      # Post-run result dialogs and image gallery
│   │
│   └── utils/                     # Shared helpers
│       ├── file_utils.py          # Directory validation, file listing, path helpers
│       ├── image_utils.py         # Image I/O, resizing, random patch cutting
│       ├── arff_writer.py         # Write features to ARFF format (Weka compatibility)
│       └── table_printer.py       # Plain-text aligned table output
│
└── external/
    ├── data/                      # Daimler benchmark data
    │   ├── 1/ 2/ 3/              # Training sets (ped_examples/ & non-ped_examples/)
    │   ├── T1/ T2/               # Test sets
    │   ├── reference_predictions.txt
    │   └── test_image_list.txt
    ├── demo_output/               # Pre-generated demo results
    ├── output/                    # Default training/prediction output
    ├── test_output/               # Output from test runs
    └── test_output_cut/           # Output from test runs with cut sampling
```

---

## 4. Algorithm

### Training

1. **Compute feature vector length** — Greyscale: 18×36 = 648 dimensions. HOG: depends on block/cell configuration (e.g., 12×12 blocks, 6×6 cells → 36 dims).
2. **Compute total non-pedestrian samples** — `non_ped_train_data_size × cuts_per_image`.
3. **Allocate** training matrix `(N, D)` and label vector `(N,)`.
4. **Read pedestrian images** (18×36 PGM), extract features, label as `+1`.
5. **Read non-pedestrian images** using one of two sampling modes:
   - **Cut**: Randomly cut `cuts` patches of size `(18×factor, 36×factor)` from the image, resize each to 18×36.
   - **Scaled**: Resize the entire image to 18×36.
6. **Write** features to an `.arff` file (for optional Weka analysis).
7. **Fit** the classifier via `GaussianNB.fit(X, y)`.

### Prediction (multi-scale sliding window)

1. **Collect** test image paths (from a directory + glob pattern, or from a file list).
2. **For each test image**:
   - Slide a detection window at scale factors `[1, 2, 3, 4, 5, 6]` — window size = `18×f` by `36×f`.
   - Step size depends on effort level: **Low** = `[18]`, **High** = `[6, 12, 18]`.
   - Resize each candidate patch to 18×36, extract features, classify with the trained model.
   - Collect all rectangles where the classifier predicts "pedestrian".
3. **Unify** overlapping detections — remove smaller rectangles fully contained within larger ones.
4. **Compare** predictions against a reference, using one of four modes:
   | Mode | Reference Source |
   |------|-----------------|
   | 0 | Entire image assumed to contain a pedestrian |
   | 1 | Image assumed to contain no pedestrian |
   | 2 | OpenCV's built-in HOG people detector (`detectMultiScale`) |
   | 3 | Rectangle coordinates read from a reference file |
5. **Compute** TP / TN / FP / FN per file and overall.

### Evaluation

- **Accuracy** = (TP + TN) / Total
- Generates three output files:
  - `prediction_summary.txt` — overall counts and accuracy
  - `confusion_matrix.txt` — image-level and rectangle-level matrices
  - `per_file_details.txt` — per-image breakdown of examined/detected rectangles

---

## 5. Architecture

The codebase separates concerns into three layers: **core** (domain logic), **gui** (presentation), and **utils** (shared helpers). Training and prediction run on background threads to keep the GUI responsive, with progress communicated back via `root.after()` callbacks.

### 5.1 Configuration (`core/config.py`)

Central module defining all constants, options, and data containers:

- **`IMG_WIDTH = 18`, `IMG_HEIGHT = 36`** — standard patch dimensions from the Daimler benchmark.
- **`PED_CLASS = 1`, `NON_PED_CLASS = -1`** — class labels used throughout.
- **`TrainOptions`** — dataclass controlling training: data paths and sizes, feature type (Greyscale / HOG), HOG block/cell sizes, non-pedestrian sampling mode (cut vs. scaled), output file names.
- **`PredictOptions`** — dataclass controlling prediction: test image source, effort level, display settings, reference comparison mode, output paths.
- **`FilePredictionDetails`** — per-file prediction results (TP/TN/FP/FN counts, predicted/reference flags).
- **`SummaryData`** — aggregated prediction statistics with an `accuracy()` method.
- **`RefPrediction`** — per-file reference rectangle data for comparison mode 3.

### 5.2 Feature Extraction (`core/feature_extraction.py`)

Stateless functions for converting images to feature vectors:

- **`extract_greyscale_features(img)`** — converts to greyscale, resizes to 18×36, flattens to a 648-dim float32 vector.
- **`extract_hog_features(img, block_size, cell_size)`** — computes a HOG descriptor using OpenCV's `HOGDescriptor` with configurable block/cell sizes and 9 orientation bins.
- **`extract_feature_vector(img, feature_vec_type, ...)`** — unified entry point that dispatches to greyscale or HOG based on the feature type parameter.
- **`compute_feature_vector_size(feature_vec_type, hog_block_size)`** — returns the expected feature dimension for a given configuration.

### 5.3 Training Pipeline (`core/trainer.py`)

**`PedestrianTrainer`** orchestrates the full training workflow. Its single public method `train()` returns a fitted `GaussianNB` classifier. Internally it:

1. Determines feature vector size and allocates the data matrix.
2. Reads pedestrian images, extracts features, stores with label `+1`.
3. Reads non-pedestrian images (using cut or scaled sampling), stores with label `-1`.
4. Writes an ARFF file for external analysis.
5. Fits `GaussianNB` on the collected feature matrix and labels.
6. Reports progress and logs throughout via callbacks.

### 5.4 Prediction Pipeline (`core/predictor.py`)

**`PedestrianPredictor`** accepts a trained classifier and runs detection on test images. Its single public method `run()` iterates over all test images, executing the multi-scale sliding window search and reference comparison. Key internal behaviors:

- **Sliding window** — at each scale factor, scans the image with step sizes determined by effort level.
- **Rectangle unification** — eliminates smaller detections fully enclosed by larger ones.
- **Reference comparison** — supports four modes (all-ped, no-ped, HOG detector, file-based) for computing TP/TN/FP/FN.
- Accumulates results in `examined_rects`, `pedestrian_rects`, `pedestrian_rects_unified`, and `pred_details` dictionaries keyed by file path.

### 5.5 Evaluation (`core/evaluator.py`)

**`Evaluator`** takes a completed predictor and generates the three output reports. Its public method `write_all()` calls `create_summary()` to aggregate image-level and rectangle-level summaries, then writes prediction summaries, confusion matrices, and per-file details to text files using the table printer utility.

### 5.6 GUI Layer (`gui/`)

| Module | Class | Role |
|---|---|---|
| `app.py` | `App` | Main window (`tk.Tk`), hosts a `ttk.Notebook` with Train and Predict tabs, the output panel, and a status bar. Manages shared state (classifier, options). |
| `train_tab.py` | `TrainTab` | Form controls for data paths, sample sizes, non-ped sampling parameters, feature type, and a **Train** button. Runs training on a background thread. |
| `predict_tab.py` | `PredictTab` | Form controls for test image source, effort level, display options, reference comparison mode, and a **Detect** button. Enabled after training completes. |
| `output_panel.py` | `OutputPanel` | Scrollable text log with level-based formatting (INFO, WARNING, ERROR). |
| `output_panel.py` | `StatusBar` | Status label + `ttk.Progressbar` at the bottom of the window. |
| `result_dialogs.py` | `ResultDialog` | Post-run dialog showing success/failure and a link to open output. |
| `result_dialogs.py` | `ImageGalleryDialog` | Paginated thumbnail viewer for browsing detection result images. |

### 5.7 Utilities (`utils/`)

| Module | Purpose |
|---|---|
| `file_utils.py` | `list_images()`, `dir_contains_images()`, `ensure_dir()`, `clean_dir()`, `create_or_clean_dir()`, `read_file_list()` — directory and file operations. |
| `image_utils.py` | `read_image()`, `determine_cut_rects()`, `get_non_ped_feature_vectors()` — image I/O and random patch sampling for non-pedestrian training data. |
| `arff_writer.py` | `write_arff()` — serializes the feature matrix and labels to ARFF format for optional use with Weka. |
| `table_printer.py` | `TableCell` class + `print_table()` — plain-text aligned table rendering for report files. |

---

## 6. Dependencies

```
opencv-python>=4.5
numpy>=1.21
scikit-learn>=1.0
Pillow>=9.0
```

**tkinter** is included with standard Python on all platforms — no extra install needed.
