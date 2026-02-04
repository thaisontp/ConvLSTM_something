import torch
import torch.nn as nn
import math

# --- CLASS 1: KHỐI TRÍCH XUẤT ĐẶC TRƯNG CNN ---
class cnnlayers(nn.Module):
    def __init__(self, input_dim=10, seq_length=100):
        super(cnnlayers, self).__init__()
        
        # Lớp CNN 1: Trích xuất đặc trưng thô
        # Lớp 1: Kernel 7 - Để học các đặc trưng chu kỳ tuần (Weekly trends)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Lớp CNN 2: Kết hợp các đặc trưng bậc thấp
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Lớp CNN 3: Trích xuất đặc trưng bậc cao (phức tạp)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Dropout để chống overfitting (như trong hình)
        self.dropout = nn.Dropout(0.3)

        # Tính toán tự động kích thước sau khi qua 3 lớp Pool
        # Với 3 lần MaxPool(2), seq_length sẽ giảm đi 8 lần
        self.final_seq_len = math.floor(seq_length / 2 / 2 / 2)
        self.flatten_size = 128 * self.final_seq_len

    def forward(self, x):
        # x: [batch, 10, 100] (10 biến, 100 bước thời gian)
        
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = self.dropout(x)
        # Chuyển đổi để phù hợp với LSTM: [batch, time, channels]
        x = x.permute(0, 2, 1)  
        return x # Trả về [batch, final_seq_len, 128]


# --- CLASS 2: KHỐI CHUỖI THỜI GIAN LSTM ---
class LSTMProcessor(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, num_layers=3, dropout_rate=0.2):
        super(LSTMProcessor, self).__init__()
        # Tích hợp 3 lớp LSTM và Dropout trực tiếp vào nn.LSTM
        # Lưu ý: dropout trong nn.LSTM chỉ áp dụng giữa các lớp (giữa lớp 1-2, 2-3)

        # Dropout trước LSTM
        self.pre_dropout = nn.Dropout(dropout_rate)

        # Xây dựng LSTM với dropout giữa các lớp
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0 
        )
        self.post_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x sau CNN cần xoay trục: [batch, channels, time] -> [batch, time, channels]
        # x: [B, T, 128] từ CNN

        x = self.pre_dropout(x)

        out, _ = self.lstm(x)      # [B, T, hidden]
        out = out[:, -1, :]        # last timestep

        out = self.post_dropout(out)

        return out                 # [B, hidden]
        # Trả về [batch, hidden_size]
        

# --- CLASS 3: KHỐI ĐẦU RA FULLY CONNECTED ---
class FinalPredictor(nn.Module):
    def __init__(self, input_size=64, output_dim=1, dropout_rate=0.2):
        super(FinalPredictor, self).__init__()
        self.fc_net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.fc_net(x)


# --- CLASS TỔNG HỢP (KẾT NỐI 3 KHỐI) ---
class SeaWaterQualityModel(nn.Module):
    def __init__(self, input_dim=10, seq_length=100, output_dim=1):
        super(SeaWaterQualityModel, self).__init__()
        
        self.cnn = cnnlayers(input_dim=input_dim, seq_length=seq_length)
        self.lstm = LSTMProcessor(input_size=128, hidden_size=64, num_layers=3)
        self.fc = FinalPredictor(input_size=64, output_dim=output_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.lstm(x)
        x = self.fc(x)
        return x