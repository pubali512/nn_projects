# Vehicle Speed Prediction (LSTM)

## Project Overview

An **LSTM-based vehicle speed prediction** system that generates synthetic vehicle sensor data, trains a recurrent neural network, and evaluates prediction accuracy. The pipeline covers the full lifecycle: physics-based data generation, preprocessing with feature engineering, LSTM model training, and evaluation with multiple visualizations.

### Main Features

- **Synthetic data generation** — Simulates vehicle physics with throttle (sinusoidal + noise), brake (random), discrete gears (1–5), drag coefficient, and mass to produce realistic sensor data
- **Feature engineering** — One-hot encodes gear values, standard-scales continuous features, and creates sliding window sequences for time-series input
- **Configurable LSTM model** — Parameterizable hidden size, number of LSTM layers, and input features
- **Training pipeline** — MSE loss with Adam optimizer, 80/20 train/test split, batch training
- **Evaluation & visualization** — Computes MSE and RMSE, generates:
  - Actual vs. predicted speed (time series overlay)
  - Prediction error over time
  - Error histogram

## Technologies Used & Installation Instructions

### Technologies

- **Python 3**
- **PyTorch** (`torch.nn`, LSTM, Adam optimizer) — Deep learning framework
- **pandas** — CSV data I/O and manipulation
- **NumPy** — Numerical computing
- **scikit-learn** (`StandardScaler`) — Feature scaling
- **matplotlib** — Visualization and plotting

### Prerequisites

- Python 3.x
- pip or conda

### Installation

```bash
cd nn_projects/vehicle_speed_prediction

# Install dependencies
pip install torch pandas numpy scikit-learn matplotlib
```

## Usage Instructions

### Running

```bash
python vehicleSpeedPrediction.py
```

The script will:
1. Generate `vehicle_sensor_data.csv` with 5,000 synthetic sensor readings
2. Preprocess features (one-hot encoding, scaling, windowing)
3. Train the LSTM model for 150 epochs
4. Evaluate on the test set and display MSE/RMSE metrics
5. Show three evaluation plots: actual vs. predicted speed, error over time, and error histogram

### Default Configuration

| Parameter          | Default Value |
|--------------------|---------------|
| Samples            | 5,000         |
| Time step (dt)     | 0.1 s         |
| Sensor noise (σ)   | 0.3           |
| Sequence length    | 100           |
| Hidden size        | 128           |
| LSTM layers        | 2             |
| Learning rate      | 0.0005        |
| Epochs             | 150           |
| Batch size         | 128           |
| Train/test split   | 80/20         |
| Random seed        | 42            |

### Physics Simulation Parameters

The synthetic data generator models:
- **Engine force** — Based on throttle input (sinusoidal pattern + Gaussian noise)
- **Brake force** — Random braking events
- **Gear shifting** — Discrete gears (1–5) based on current speed thresholds
- **Aerodynamic drag** — Proportional to speed squared
- **Vehicle mass** — Constant parameter affecting acceleration

### Files

| File                        | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| `vehicleSpeedPrediction.py` | Full pipeline: data generation, preprocessing, training, evaluation |
| `vehicle_sensor_data.csv`   | Generated synthetic data (5,001 rows including header). Columns: `time`, `throttle`, `brake`, `gear`, `speed` |
| `icons.py`                  | Console output formatting utility (emoji + color codes)         |
