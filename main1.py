import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# 1. Load Data
data = pd.read_csv('China-historical-2016-en.csv')

# Định nghĩa Target (Y) và Features (X)
target_cols = ['pH','Dissolved Oxygen Saturation (%)','Dissolved Oxygen (mg/L)',
               'Nitrate Nitrogen (mg/L)','Total Nitrogen (mg/L)','Temperature (deg C)']

# Các cột không dùng để train nhưng cần để quản lý
meta_cols = ['Date', 'Station'] 
drop_cols = ['Water Control Zone', 'Source','Sample No','Depth'] # Cột bỏ đi

# 2. Xử lý làm sạch (Clean Value) - Tách riêng ra để dễ kiểm soát
def clean_value_logic(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().lower()
    if x in ["nd", "na", "-", ""]: return np.nan
    
    # Xử lý <LOD
    if x.startswith("<"):
        try:
            lod = float(x.replace("<", "").strip())
            return lod / math.sqrt(2) # Quy ước thay thế LOD
        except: return np.nan
        
    # Xử lý số thường
    x = re.sub(r"[^\d\.]", "", x)
    try: return float(x)
    except: return np.nan

# Áp dụng clean cho các cột dữ liệu số (Trừ Date và Station)
numeric_cols = [c for c in data.columns if c not in meta_cols + drop_cols]
for col in numeric_cols:
    data[col] = data[col].apply(clean_value_logic)

# 3. Sắp xếp dữ liệu quan trọng nhất: Station -> Date
# Convert Date sang datetime để sort đúng
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce') # Check lại format date trong csv của bạn
data = data.sort_values(by=['Station', 'Date']).reset_index(drop=True)

# 4. Impute (CHƯA scale để tránh data leakage)
# Lưu ý: Nên Impute theo từng trạm sẽ chính xác hơn, nhưng để đơn giản ta làm gộp trước
feature_cols = [c for c in data.columns if c not in drop_cols + meta_cols] # Chỉ lấy cột số liệu

# Tách X, Y
data_values = data[feature_cols].values

# Impute (fit trên toàn bộ data OK vì chỉ là fill missing values)
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_values)

# Đưa lại vào DataFrame để dễ xử lý group
df_processed = pd.DataFrame(data_imputed, columns=feature_cols)
df_processed['Station'] = data['Station'] # Trả lại cột Station để group

# 5. Hàm tạo Sequence thông minh (theo từng Station)
def create_sequences_by_station(df, lookback=10, target_columns=target_cols):
    X_seq, y_seq = [], []
    
    # Group by Station để không bị trượt cửa sổ qua trạm khác
    for station_name, group in df.groupby('Station'):
        group_values = group.drop(['Station'], axis=1).values # Bỏ cột Station đi, chỉ lấy số liệu
        
        if len(group_values) <= lookback: continue # Bỏ qua trạm quá ít dữ liệu
            
        for i in range(len(group_values) - lookback):
            # X: Lấy từ i đến i+lookback (bao gồm cả các cột target trong quá khứ để làm feature)
            X_seq.append(group_values[i : i + lookback])
            
            # y: Lấy giá trị tại thời điểm i + lookback (chỉ lấy cột target)
            # Cần map index của target columns trong feature_cols
            target_indices = [df.columns.get_loc(c) for c in target_columns]
            y_seq.append(group_values[i + lookback, target_indices])
            
    return np.array(X_seq), np.array(y_seq)

lookback = 10
X, y = create_sequences_by_station(df_processed, lookback=lookback, target_columns=target_cols)

print(f"X Shape trước scale: {X.shape}") # (Samples, 10, n_features)
print(f"y Shape: {y.shape}") # (Samples, 6)

# 6. Split Train/Validation/Test TRƯỚC scaling (60/20/20)
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)

X_train = X[:train_size].copy()
X_val = X[train_size:train_size + val_size].copy()
X_test = X[train_size + val_size:].copy()

y_train = y[:train_size].copy()
y_val = y[train_size:train_size + val_size].copy()
y_test = y[train_size + val_size:].copy()

print(f"Train/Val/Test Split:")
print(f"  Train: {len(X_train)} samples")
print(f"  Valid: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# 7. Scale - FIT CHỈ TRÊN TRAINING DATA để tránh data leakage
# Reshape để fit scaler (scaler cần 2D array)
X_train_2d = X_train.reshape(-1, X_train.shape[-1])
X_val_2d = X_val.reshape(-1, X_val.shape[-1])
X_test_2d = X_test.reshape(-1, X_test.shape[-1])

scaler = StandardScaler()
X_train_scaled_2d = scaler.fit_transform(X_train_2d)  # Fit chỉ trên train
X_val_scaled_2d = scaler.transform(X_val_2d)         # Transform validation dùng thống kê từ train
X_test_scaled_2d = scaler.transform(X_test_2d)        # Transform test dùng thống kê từ train

# Reshape lại thành 3D
X_train = X_train_scaled_2d.reshape(X_train.shape)
X_val = X_val_scaled_2d.reshape(X_val.shape)
X_test = X_test_scaled_2d.reshape(X_test.shape)

# Transpose để phù hợp với PyTorch Conv1D: (batch_size, sequence_length, channels) -> (batch_size, channels, sequence_length)
X_train = np.transpose(X_train, (0, 2, 1))  # From (batch, 10, channels) to (batch, channels, 10)
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

print(f"X_train Shape sau scale (PyTorch Conv1D format): {X_train.shape}")  # (batch_size, n_features, 10)
print(f"X_val Shape sau scale (PyTorch Conv1D format): {X_val.shape}")
print(f"X_test Shape sau scale (PyTorch Conv1D format): {X_test.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_val Shape: {y_val.shape}")
print(f"y_test Shape: {y_test.shape}")
# print("Data ready for Conv1D - NO DATA LEAKAGE!")

# 8. Chuyển dữ liệu sang Tensor để dùng trong PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
print("Data converted to PyTorch Tensors.")

# Mẫu mô hình Conv1D output 3D tensor cho LSTM
class CNN1D(nn.Module):
    def __init__(
        self,
        in_channels,        # = n_features
        out_dim,            # = số target (6) - NOT USED, kept for compatibility
        lookback,           # = sequence_length (10)
        num_filters=32,     # số kernel Conv1D
        kernel_size=3,      # kích thước kernel
        dropout=0.2
    ):
        super(CNN1D, self).__init__()

        # ----- Conv Block 1 -----
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm1d(num_filters)

        # ----- Conv Block 2 -----
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm1d(num_filters * 2)

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, sequence_length)
        Output shape: (batch_size, seq_len, num_filters*2) - 3D tensor for LSTM
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Transpose từ (batch, channels, seq_len) → (batch, seq_len, channels)
        # Để compatible với LSTM input (batch_first=True)
        x = x.transpose(1, 2)
        return x

# Khởi tạo mô hình
n_features = X_train.shape[1]   # in_channels
lookback   = X_train.shape[2]
n_targets  = y_train.shape[1]

model = CNN1D(
    in_channels=n_features,
    out_dim=n_targets,
    lookback=lookback,
    num_filters=32,
    kernel_size=3
)

print(model)

# LSTM model để tham khảo kết hợp với CNN1D
class CNN_LSTM_WaterQuality(nn.Module):
    def __init__(self, cnn_module, n_targets, hidden_size=128, num_layers=1):
        super(CNN_LSTM_WaterQuality, self).__init__()
        self.cnn = cnn_module
        
        # input_size của LSTM = out_channels của lớp conv cuối cùng (64)
        # Chúng ta chọn batch_first=True vì tensor của bạn có dạng (Batch, Seq, Feature)
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Lớp Fully Connected để chuyển từ hidden state sang 6 chỉ số đầu ra
        self.fc = nn.Linear(hidden_size, n_targets)

    def forward(self, x):
        # 1. Qua CNN: (batch, 24, 10) -> (batch, 64, 6)
        x = self.cnn(x)
        
        # 2. Qua LSTM: 
        # Cần transpose về (batch, seq=6, feature=64) đã làm trong class CNN1D của bạn
        # lstm_out: (batch, 6, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 3. Lấy Hidden state của timestep cuối cùng (tức là bước thứ 6)
        # hn shape: (num_layers, batch, hidden_size)
        # Ở đây ta lấy hn[-1] hoặc lstm_out[:, -1, :]
        last_step_out = lstm_out[:, -1, :]
        
        # 4. Dự đoán đầu ra
        out = self.fc(last_step_out)
        return out

# Khởi tạo model
n_targets = y_train_tensor.shape[1] # 6
final_model = CNN_LSTM_WaterQuality(model, n_targets=n_targets, hidden_size=128)

print(final_model)

# # Kiểm tra forward pass với batch mẫu
# sample_input = X_train_tensor[:8]  # Lấy 8 mẫu đầu tiên làm batch thử   
# sample_output = final_model(sample_input)
# print(f"Sample input shape: {sample_input.shape}")   # (8, n_features
# print(f"Sample output shape: {sample_output.shape}") # (8, 6)
# # Output shape sẽ là (batch_size, 6) tương ứng với 6 chỉ số đầu ra
# print("Forward pass successful with sample input.")
# print(f"Sample input shape: {sample_input.shape}")   # (8, n_features, 10)
# print(f"Sample output shape: {sample_output.shape}") # (8, 6)
# # Output shape sẽ là (batch_size, 6) tương ứng với 6 chỉ số đầu ra
# print("Forward pass successful with sample input.")

# 9. Tạo DataLoader cho Train/Validation/Test
# ============================================
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Validation DataLoader: {len(val_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")
print(f"Batch size: {batch_size}")

# # 10. Setup Training
# # ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# QUAN TRỌNG: Phải dùng final_model (CNN + LSTM) thay vì model (chỉ CNN)
# Chuyển model tổng hợp lên GPU/CPU
final_model = final_model.to(device) 

# Sử dụng tham số của final_model cho Optimizer
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

print("\n" + "="*60)
print("Testing forward pass with real training batch...")
print("="*60)

# Lấy batch đầu tiên từ train_loader để test
for X_batch, y_batch in train_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    
    print(f"Batch X shape: {X_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    
    # Forward pass
    y_pred = final_model(X_batch)
    print(f"Model output shape: {y_pred.shape}")
    
    # Tính loss
    loss = criterion(y_pred, y_batch)
    print(f"Loss: {loss.item():.6f}")
    
    print("Forward pass with real batch successful! ✓")
    break  # Chỉ test batch đầu tiên

# ============================================
# 11. Visualization TRƯỚC Training
# ============================================
print("\n" + "="*60)
print("Plotting data visualization BEFORE training...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Distribution of 6 Target Features (Before Training)', fontsize=16, fontweight='bold')

for idx, col in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]
    ax.hist(y_train[:, idx], bins=30, alpha=0.6, label='Train', color='blue')
    ax.hist(y_test[:, idx], bins=30, alpha=0.6, label='Test', color='orange')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_distribution_before_training.png', dpi=100, bbox_inches='tight')
print("✓ Saved: data_distribution_before_training.png")
plt.close()

# Visualize một sample sequence
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Sample Sequence - First 10 Features Over Time (Lookback=10)', fontsize=16, fontweight='bold')

sample_idx = 0
sample_x = X_train[sample_idx]  # Shape: (n_features, 10)

for feat_idx in range(min(6, n_features)):
    ax = axes[feat_idx // 3, feat_idx % 3]
    ax.plot(sample_x[feat_idx], marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Scaled Value')
    ax.set_title(f'Feature {feat_idx}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_sequence_visualization.png', dpi=100, bbox_inches='tight')
print("✓ Saved: sample_sequence_visualization.png")
plt.close()

# ============================================
# 12. Training Loop với Loss Tracking
# ============================================
print("\n" + "="*60)
print("Starting Training...")
print("="*60)

epochs = 220
train_losses = []
val_losses = []
test_losses = []
best_val_loss = float('inf')
early_stopping_patience = 50
patience_counter = 0

for epoch in range(epochs):
    # ===== TRAIN PHASE =====
    final_model.train()
    train_loss_epoch = 0
    n_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = final_model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss_epoch += loss.item()
        n_batches += 1
    
    train_loss_epoch /= n_batches
    train_losses.append(train_loss_epoch)
    
    # ===== VALIDATION PHASE =====
    final_model.eval()
    val_loss_epoch = 0
    n_val_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = final_model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            val_loss_epoch += loss.item()
            n_val_batches += 1
    
    val_loss_epoch /= n_val_batches
    val_losses.append(val_loss_epoch)
    
    # ===== TEST PHASE (chỉ đánh giá, không tối ưu) =====
    test_loss_epoch = 0
    n_test_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = final_model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            test_loss_epoch += loss.item()
            n_test_batches += 1
    
    test_loss_epoch /= n_test_batches
    test_losses.append(test_loss_epoch)
    
    # Early Stopping based on Validation Loss
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        patience_counter = 0  # Reset counter
        torch.save(final_model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss_epoch:.6f} | Val Loss: {val_loss_epoch:.6f} | Test Loss: {test_loss_epoch:.6f} ✓ (Best)")
    else:
        patience_counter += 1
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss_epoch:.6f} | Val Loss: {val_loss_epoch:.6f} | Test Loss: {test_loss_epoch:.6f}")
        
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly Stopping at Epoch {epoch+1} (patience reached)")
            break

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.6f}")

# Load best model
final_model.load_state_dict(torch.load('best_model.pth'))

# ============================================
# 13. Predictions on Train & Test Set
# ============================================
print("\n" + "="*60)
print("Making predictions...")
print("="*60)

final_model.eval()
y_pred_train_list = []
y_pred_val_list = []
y_pred_test_list = []

with torch.no_grad():
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        y_pred = final_model(X_batch)
        y_pred_train_list.append(y_pred.cpu().numpy())
    
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        y_pred = final_model(X_batch)
        y_pred_val_list.append(y_pred.cpu().numpy())
    
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_pred = final_model(X_batch)
        y_pred_test_list.append(y_pred.cpu().numpy())

y_pred_train = np.concatenate(y_pred_train_list, axis=0)
y_pred_val = np.concatenate(y_pred_val_list, axis=0)
y_pred_test = np.concatenate(y_pred_test_list, axis=0)

print(f"y_pred_train shape: {y_pred_train.shape}")
print(f"y_pred_val shape: {y_pred_val.shape}")
print(f"y_pred_test shape: {y_pred_test.shape}")

# ============================================
# 14. Visualization AFTER Training
# ============================================
print("\nPlotting results...")

# Plot 1: Loss Curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_losses, label='Train Loss', linewidth=2, marker='o', markersize=3)
ax.plot(val_losses, label='Validation Loss', linewidth=2, marker='^', markersize=3)
ax.plot(test_losses, label='Test Loss', linewidth=2, marker='s', markersize=3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('Loss Curve During Training (Train/Val/Test)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=100, bbox_inches='tight')
print("✓ Saved: loss_curve.png")
plt.close()

# Plot 2: True vs Predicted - Test Set
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('True vs Predicted Values - TEST SET', fontsize=16, fontweight='bold')

for idx, col in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]
    
    # Plot scatter
    ax.scatter(y_test[:, idx], y_pred_test[:, idx], alpha=0.6, s=20)
    
    # Plot perfect prediction line
    min_val = min(y_test[:, idx].min(), y_pred_test[:, idx].min())
    max_val = max(y_test[:, idx].max(), y_pred_test[:, idx].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Value', fontsize=10)
    ax.set_ylabel('Predicted Value', fontsize=10)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('true_vs_predicted_test.png', dpi=100, bbox_inches='tight')
print("✓ Saved: true_vs_predicted_test.png")
plt.close()

# Plot 3: Residuals (Error Distribution)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Residuals (Error) Distribution - TEST SET', fontsize=16, fontweight='bold')

residuals = y_test - y_pred_test

for idx, col in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]
    ax.hist(residuals[:, idx], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Error (True - Predicted)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_distribution.png', dpi=100, bbox_inches='tight')
print("✓ Saved: residuals_distribution.png")
plt.close()

# ============================================
# 15. Performance Metrics
# ============================================

print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)

# Train metrics
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

# Validation metrics
val_mse = mean_squared_error(y_val, y_pred_val)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_pred_val)
val_r2 = r2_score(y_val, y_pred_val)

# Test metrics
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\nTRAIN SET:")
print(f"  MSE: {train_mse:.6f}")
print(f"  RMSE: {train_rmse:.6f}")
print(f"  MAE: {train_mae:.6f}")
print(f"  R² Score: {train_r2:.6f}")

print("\nVALIDATION SET:")
print(f"  MSE: {val_mse:.6f}")
print(f"  RMSE: {val_rmse:.6f}")
print(f"  MAE: {val_mae:.6f}")
print(f"  R² Score: {val_r2:.6f}")

print("\nTEST SET:")
print(f"  MSE: {test_mse:.6f}")
print(f"  RMSE: {test_rmse:.6f}")
print(f"  MAE: {test_mae:.6f}")
print(f"  R² Score: {test_r2:.6f}")

# Plot 4: Metrics Comparison
fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['MSE', 'RMSE', 'MAE', 'R² Score']
train_vals = [train_mse, train_rmse, train_mae, train_r2]
val_vals = [val_mse, val_rmse, val_mae, val_r2]
test_vals = [test_mse, test_rmse, test_mae, test_r2]

x = np.arange(len(metrics))
width = 0.25

ax.bar(x - width, train_vals, width, label='Train', alpha=0.8)
ax.bar(x, val_vals, width, label='Validation', alpha=0.8)
ax.bar(x + width, test_vals, width, label='Test', alpha=0.8)

ax.set_ylabel('Value', fontsize=12)
ax.set_title('Model Performance Metrics (Train/Validation/Test)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved: metrics_comparison.png")
plt.close()

print("\n" + "="*60)
print("All visualizations saved!")
print("="*60)

# ============================================
# PLOT: TRUE vs PREDICTED (SCATTER)
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('True vs Predicted Values (TEST SET)', fontsize=16, fontweight='bold')

for i, col in enumerate(target_cols):
    ax = axes[i // 3, i % 3]

    # Scatter plot
    ax.scatter(
        y_test[:, i],
        y_pred_test[:, i],
        alpha=0.6,
        s=20
    )

    # Đường y = x (perfect prediction)
    min_val = min(y_test[:, i].min(), y_pred_test[:, i].min())
    max_val = max(y_test[:, i].max(), y_pred_test[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('true_vs_predicted_test.png', dpi=120, bbox_inches='tight')
plt.show()
