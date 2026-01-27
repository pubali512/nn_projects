# ============================================================
# VEHICLE SPEED PREDICTION WITH LSTM MODEL (SYNTHETIC DATA)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler                # StandardScaler: Improves LSTM & NN accuracy
import matplotlib.pyplot as plt
import random
import icons

# =============================================
# Utility: Set seed(Random Number Generator)
# =============================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Generate CSV (Synthetic Vehicle Data)
# =========================
def generate_csv(filename, samples, dt, noise_std): 
    time = np.arange(samples) * dt
    
    # -----------------------   
    # Driver inputs
    # -----------------------
    throttle = np.clip(
        0.6 * np.sin(0.02 * time) + 0.4 * np.random.randn(samples),
        0, 1
    )
    
    brake = np.clip(0.2 * np.random.randn(samples), 0, 1)
    
    # -----------------------   
    # Discrete gears
    # -----------------------
    gear = np.random.choice(
        [1, 2, 3, 4, 5],
        size=samples,
        p=[0.2, 0.25, 0.25, 0.2, 0.1]
    )

    # -----------------------   
    # Vehicle parameters
    # -----------------------
    mass = 1500.0          # kg
    engine_force = 4000.0  # N
    brake_force = 5000.0   # N
    drag_coeff = 0.32

    speed = np.zeros(samples)

    for t in range(1, samples):
        traction = engine_force * throttle[t] / gear[t]
        braking = brake_force * brake[t]
        drag = drag_coeff * speed[t-1] ** 2

        acceleration = (traction - braking - drag) / mass
        speed[t] = max(speed[t-1] + acceleration * dt, 0)
        
    # -----------------------
    # Sensor noise
    # -----------------------
    speed += np.random.normal(0, noise_std, samples)

    # -----------------------
    # Save CSV
    # -----------------------
    df = pd.DataFrame({
        "time": time,
        "throttle": throttle,
        "brake": brake,
        "gear": gear,
        "speed": speed
    })

    df.to_csv(filename, index=False)
    
    print("\n" + icons.info(f"Synthetic vehicle data saved to {filename}"))

# =========================
# Sequence Builder
# =========================
def create_sequences(X, y, seq_len):
    
    """
    Build sequences for LSTM:
    X: features (throttle, brake, gear_onehot)
    y: Δspeed (target)
    seq_len: sequence length
    """
    
    Xs, ys = [], []
    
    #delta_y = np.diff(y, prepend=y[0])
    
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])      
        ys.append(y[i + seq_len])
        #ys.append(y[i+1 : i+seq_len+1])

        
    return np.array(Xs), np.array(ys)

# ==============================================================
# LSTM Model (Long Short-Term Memory (LSTM) neural network)
# ==============================================================
class SpeedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =========================
# Trainer Class
# =========================
class VehicleSpeedTrainer:
    def __init__(self, csv_file, seq_len, hidden_size, layers, lr):
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.layers = layers
        self.lr = lr     
        self.X_scaler = StandardScaler()                    # Scale continuous features only                   
        self.y_scaler = StandardScaler()                    # Scale target variable (speed)           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("\n" + icons.device(f" Using device: {self.device}"))
        print()

    def load_data(self):
        data = pd.read_csv(self.csv_file)

        # One-hot encode 'gear' feature: Allows the model to properly learn the effect of gear shifts.
        gear_onehot = pd.get_dummies(data["gear"], prefix="gear")
        
        # Scale only continuous features (throttle, brake, speed) for better numerical stability,
        X_cont = self.X_scaler.fit_transform(data[["throttle", "brake","speed"]].values)
        X = np.hstack([X_cont, gear_onehot.values])

        # Target variable: Absolute speed
        speed = data["speed"].values
        self.speed_abs = speed 
        
        y = self.y_scaler.fit_transform(speed.reshape(-1, 1)).ravel()


        # Create sequences
        X_seq, y_seq = create_sequences(X, y, self.seq_len)

        # Split train/test
        split = int(0.8 * len(X_seq))
        self.X_train = torch.tensor(X_seq[:split], dtype=torch.float32)
        self.y_train = torch.tensor(y_seq[:split], dtype=torch.float32).view(-1, 1)
        self.X_test = torch.tensor(X_seq[split:], dtype=torch.float32)
        self.y_test = torch.tensor(y_seq[split:], dtype=torch.float32).view(-1, 1)
        
        self.X_features = X.shape[1]
        
        self.split_idx = split  # Save split index to recover correct initial absolute speed when reconstructing from Δspeed


    def train(self, epochs):
        self.model = SpeedLSTM(
            input_size=self.X_features,
            hidden_size=self.hidden_size,
            layers=self.layers
        )

        self.model.to(self.device)
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(icons.steps(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}"))

        print("\n" + icons.info(" Training complete."))
        print()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            # Model output
            preds = self.model(self.X_test)
            
            # Move to CPU + NumPy for evaluation/plotting
            preds = preds.cpu().numpy()
            y_true = self.y_test.cpu().numpy()

            # Inverse scaling (y_scaler only)
            preds = self.y_scaler.inverse_transform(preds)
            y_true = self.y_scaler.inverse_transform(y_true)
            
            # Use raw model outputs as absolute speeds
            pred_speed = preds
            true_speed = y_true

            # Clip absolute speed 
            pred_speed = np.clip(pred_speed, 0, 40)                        # Vehicle speed can't be negative, max ~40 m/s
            
            # Check first few predictions vs ground truth
            print("\n" + icons.check(f"First 5 true speeds: {true_speed[:5].ravel()}"))
            print(icons.check(f"First 5 predicted speeds: {pred_speed[:5].ravel()}"))
            
            # Compute metrics
            mse_loss = np.mean((pred_speed - true_speed) ** 2)             # MSE: Mean Squared Error
            rmse_loss = np.sqrt(mse_loss)                                  # RMSE: Root Mean Squared Error
            
            # Check min/max values of predictions and ground truth 
            print("\n" + icons.check(f"Preds min/max: {pred_speed.min():.3f}, {pred_speed.max():.3f}"))
            print(icons.check(f"y_true min/max: {true_speed.min():.3f}, {true_speed.max():.3f}"))
            
            # Display Mean Squared Error and Root Mean Squared Error for test predictions   
            print("\n" + icons.info(f"Test MSE: {mse_loss:.3f}"))
            print("\n" + icons.info(f"Test RMSE: {rmse_loss:.3f}"))
            print()
            
        # Plot results
        plt.figure(figsize=(10, 4))
        plt.plot(true_speed[:500], label="Actual Speed", linewidth=2)
        plt.plot(pred_speed[:500], label="Predicted Speed", linestyle="--")
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Speed (m/s)")
        plt.title("Vehicle Speed Prediction (LSTM)")
        plt.show()
        
        #Error prediction
        error = pred_speed - true_speed
        
        # Plot time-series of errors
        plt.figure(figsize=(10,4))
        plt.plot(error)
        plt.title("Prediction Error Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Prediction Error (m/s)")
        plt.show()

        # Plot histogram of errors
        plt.figure(figsize=(10,4))
        plt.hist(error, bins=50)
        plt.title("Histogram of Prediction Errors")
        plt.xlabel("Prediction Error (m/s)")
        plt.ylabel("Frequency")
        plt.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    seed = 42
    samples = 5000
    delta_time = 0.1
    noise_standard = 0.3
    epochs = 150
    seq_len = 100
    hidden_size = 128
    layers = 2
    lr = 0.0005
    csv_file = "vehicle_sensor_data.csv"
    

    set_seed(seed)

    generate_csv(csv_file, samples, delta_time, noise_standard)

    trainer = VehicleSpeedTrainer(
        csv_file = csv_file,
        seq_len = seq_len,
        hidden_size = hidden_size,
        layers = layers,
        lr = lr
    )

    trainer.load_data()
    trainer.train(epochs)
    trainer.evaluate()



