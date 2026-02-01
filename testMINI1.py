import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import math
import re

# ==========================================
# 1. LOAD VÀ CLEAN DATA
# ==========================================
data = pd.read_csv('China-historical-2016-en.csv')

# Danh sách các cột mục tiêu (Dự đoán đa biến)
target_cols = ['pH', 'Dissolved Oxygen (mg/L)', 'Nitrate Nitrogen (mg/L)', 
               'Total Nitrogen (mg/L)', 'Temperature (deg C)']

meta_cols = ['Date', 'Station']
drop_cols = ['Water Control Zone', 'Source', 'Sample No', 'Depth', 'Dissolved Oxygen Saturation (%)']

def clean_value_logic(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().lower()
    if x in ["nd", "na", "-", ""]: return np.nan
    if x.startswith("<"):
        try:
            lod = float(x.replace("<", "").strip())
            return lod / math.sqrt(2)
        except: return np.nan
    x = re.sub(r"[^\d\.]", "", x)
    try: return float(x)
    except: return np.nan

# Áp dụng làm sạch cho các cột số
for col in target_cols:
    data[col] = data[col].apply(clean_value_logic)

# Xử lý thời gian và sắp xếp
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data = data.sort_values(by=['Station', 'Date']).reset_index(drop=True)

# Impute dữ liệu thiếu
imputer = SimpleImputer(strategy='mean')
data[target_cols] = imputer.fit_transform(data[target_cols])

# ==========================================
# 2. CHUẨN HÓA VÀ TẠO SEQUENCE
# ==========================================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[target_cols])
df_scaled = pd.DataFrame(data_scaled, columns=target_cols)
df_scaled['Station'] = data['Station']

def create_multivariate_sequences(df, feature_cols, seq_length):
    xs, ys = [], []
    for station in df['Station'].unique():
        station_data = df[df['Station'] == station][feature_cols].values
        if len(station_data) > seq_length:
            for i in range(len(station_data) - seq_length):
                x = station_data[i : (i + seq_length), :]
                y = station_data[i + seq_length, :] # Dự đoán tất cả các biến ở bước tiếp theo
                xs.append(x)
                ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 10 # Độ dài chuỗi thời gian
X, y = create_multivariate_sequences(df_scaled, target_cols, SEQ_LENGTH) # Tạo chuỗi đa biến

# Chia Train/Test (80/20)
split = int(len(X) * 0.8)
trainX = torch.tensor(X[:split], dtype=torch.float32)
trainY = torch.tensor(y[:split], dtype=torch.float32)
testX = torch.tensor(X[split:], dtype=torch.float32)
testY = torch.tensor(y[split:], dtype=torch.float32)

# ==========================================
# 3. ĐỊNH NGHĨA MÔ HÌNH MULTIVARIATE LSTM
# ==========================================
class WaterQualityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(WaterQualityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Lấy hidden state của bước thời gian cuối cùng
        out = self.fc(out[:, -1, :]) 
        return out

num_features = len(target_cols)
model = WaterQualityLSTM(input_dim=num_features, hidden_dim=128, layer_dim=2, output_dim=num_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. HUẤN LUYỆN
# ==========================================
epochs = 600
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(trainX)
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ==========================================
# 5. DỰ ĐOÁN VÀ INVERSE TRANSFORM
# ==========================================
model.eval()
with torch.no_grad():
    predictions = model(testX).numpy()

# Đưa dữ liệu về đơn vị gốc
actual_y = scaler.inverse_transform(testY.numpy())
predicted_y = scaler.inverse_transform(predictions)

# ==========================================
# 6. VẼ BIỂU ĐỒ SO SÁNH (Từng chỉ số)
# ==========================================
fig, axes = plt.subplots(nrows=len(target_cols), ncols=1, figsize=(12, 18))
plt.subplots_adjust(hspace=0.4)

for i, col in enumerate(target_cols):
    axes[i].plot(actual_y[:200, i], label='Thực tế', color='blue', alpha=0.7)
    axes[i].plot(predicted_y[:200, i], label='Dự đoán', color='red', linestyle='--')
    axes[i].set_title(f'Dự báo chỉ số: {col}')
    axes[i].set_ylabel('Giá trị gốc')
    axes[i].legend()

plt.xlabel('Mẫu thử nghiệm (200 điểm đầu tiên trong tập Test)')
plt.show()