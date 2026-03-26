import wfdb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 1. CẤU HÌNH & ÁNH XẠ NHÃN (AAMI EC57)
# ==========================================
AAMI_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', # Normal
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',           # Supraventricular
    'V': 'V', 'E': 'V',                               # Ventricular
    'F': 'F',                                         # Fusion
    '/': 'Q', 'f': 'Q', 'Q': 'Q'                      # Unknown
}
NON_BEAT_ANNOTATIONS = ['~', '|', '+', 's', 't', 'u', 'p', '^', '`', '"', '@', 'x']

# Chuyển đổi nhãn AAMI sang số nguyên (0-4) để Pytorch hiểu
LABEL_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

FS = 360  # Tần số lấy mẫu của MIT-BIH
WINDOW_LEFT = int(0.2 * FS)   # 0.2s trước R-peak ~ 72 mẫu
WINDOW_RIGHT = int(0.4 * FS)  # 0.4s sau R-peak ~ 144 mẫu
WINDOW_SIZE = WINDOW_LEFT + WINDOW_RIGHT # 216 mẫu cho mỗi nhịp tim

# ==========================================
# 2. HÀM TRÍCH XUẤT DỮ LIỆU TỪ PHYSIONET
# ==========================================
def extract_heartbeats(record_names, data_dir='mitdb', max_records=5):
    """
    Trích xuất ra từng nhịp tim (cắt cửa sổ quanh R-peak) từ một danh sách bản ghi.
    Trả về dữ liệu X (tín hiệu 1D) và y (nhãn 0-4).
    """
    X, y = [], []
    print(f"Bắt đầu trích xuất dữ liệu từ {min(len(record_names), max_  records)} bản ghi...")
    
    for record_name in tqdm(record_names[:max_records]):
        try:
            # Tải bản ghi và chú thích
            record = wfdb.rdrecord(record_name, pn_dir=data_dir)
            annotation = wfdb.rdann(record_name, 'atr', pn_dir=data_dir)
            
            # Lấy tín hiệu kênh đầu tiên (thường là MLII)
            signal = record.p_signal[:, 0]
            
            for symbol, sample_idx in zip(annotation.symbol, annotation.sample):
                if symbol in NON_BEAT_ANNOTATIONS:
                    continue
                
                # Kiểm tra xem cửa sổ cắt có vượt quá giới hạn mảng tín hiệu không
                start_idx = sample_idx - WINDOW_LEFT
                end_idx = sample_idx + WINDOW_RIGHT
                
                if start_idx < 0 or end_idx >= len(signal):
                    continue
                
                # Ánh xạ nhãn sang AAMI
                aami_label = AAMI_MAPPING.get(symbol, 'Q')
                label_id = LABEL_TO_ID[aami_label]
                
                # Cắt đoạn tín hiệu nhịp tim
                beat_signal = signal[start_idx:end_idx]
                
                X.append(beat_signal)
                y.append(label_id)
                
        except Exception as e:
            print(f"Lỗi khi xử lý bản ghi {record_name}: {e}")
            
    return np.array(X), np.array(y)

# Danh sách các bản ghi chuẩn của MIT-BIH
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214',
    '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
]

# ==========================================
# 3. ĐỊNH NGHĨA DATASET & MÔ HÌNH CNN (PYTORCH)
# ==========================================
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # Thêm chiều kênh (N, 1, L)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HeartbeatCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(HeartbeatCNN, self).__init__()
        
        # Cấu trúc mạng 1D CNN cơ bản và hiệu quả cho chuỗi thời gian
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # Tính toán kích thước đầu ra sau các lớp Pooling:
        # Input: 216 -> MP1: 108 -> MP2: 54 -> MP3: 27
        # 128 channels * 27 = 3456
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 27, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def plot_sample_heartbeats(X, y):
    """
    Vẽ 5-10 nhịp tim mẫu từ các lớp khác nhau để xem hình dạng của cửa sổ cắt.
    """
    print("\n=> Đang vẽ biểu đồ một vài mẫu nhịp tim điển hình...")
    plt.figure(figsize=(15, 10))
    time_axis = np.linspace(-0.2, 0.4, WINDOW_SIZE)
    
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    for i, cls in enumerate(unique_classes):
        # Lấy tối đa 2 mẫu đầu tiên của mỗi lớp
        indices = np.where(y == cls)[0][:2]
        
        for j, idx in enumerate(indices):
            plt.subplot(num_classes, 2, i * 2 + j + 1)
            plt.plot(time_axis, X[idx], color='b' if j==0 else 'g')
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='R-peak (0s)')
            plt.title(f'Lớp: {ID_TO_LABEL[cls]} (Mẫu {j+1})')
            plt.xlabel('Thời gian so với đỉnh R (giây)')
            plt.ylabel('Biên độ')
            plt.grid(True, linestyle=':', alpha=0.6)
            if j == 0:
                plt.legend()
                
    plt.tight_layout()
    # Hiển thị trực tiếp thay vì lưu ảnh
    plt.show()
    print("Đã hiển thị cửa sổ biểu đồ các nhịp tim mẫu.")

# ==========================================
# 4. CHỨC NĂNG HUẤN LUYỆN
# ==========================================
def train_model():
    print("--------------------------------------------------")
    print(" BÀI TOÁN PHÂN LOẠI NHỊP TIM (CNN 1D) - MIT-BIH")
    print("--------------------------------------------------\n")
    
    # Số lượng bản ghi để tải: tăng lên nêú muốn độ chính xác cao hơn, (tối đa ~48)
    # Tải 5 bản ghi ở đây cho mục đích mô phỏng nhanh
    NUM_RECORDS_TO_USE = 8  
    X, y = extract_heartbeats(MITBIH_RECORDS, max_records=NUM_RECORDS_TO_USE)
    
    print(f"\n=> Đã trích xuất {len(X)} nhịp tim.")
    
    unique_classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        print(f"  + Lớp {ID_TO_LABEL[cls]}: {count} mẫu")
    
    if len(X) == 0:
        print("Không có dữ liệu, hãy kiểm tra kết nối mạng.")
        return
        
    # Gọi hàm vẽ biểu đồ mẫu
    plot_sample_heartbeats(X, y)
    
    # Dừng chương trình tại đây để cho phép người dùng xem biểu đồ trước khi train
    print("\n[!] Chương trình tạm dừng tại đây để bạn có thể xem trước file 'heartbeat_samples.png'.")
    print("Nếu bạn muốn tiếp tục huấn luyện mô hình, hãy comment dòng 'return' này lại.")
    return
        
    # Chia dữ liệu: 80% Train, 20% Test (Phân tầng - stratify để đảm bảo chia đều các nhãn hiếm)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Tạo DataLoaders
    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Khởi tạo Mô hình, Hàm Loss (có tính trọng số vì dữ liệu mất cân bằng nặng), Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeartbeatCNN(num_classes=5).to(device)
    
    # Do dữ liệu MIT-BIH cực kỳ lệch (lớp N rất lớn), ta cần một hàm mất mát có trọng số
    # Tính toán đơn giản nghịch đảo tần suất
    weights = np.zeros(5)
    for i in range(5):
        if len(counts[unique_classes == i]) > 0:
            weights[i] = 1.0 / counts[unique_classes == i][0]
        else:
            weights[i] = 0.0 # Bỏ qua lớp không có mẫu
            
    weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Huấn luyện (Training Loop)
    EPOCHS = 10
    print(f"\nBắt đầu huấn luyện mô hình trên {device} trong {EPOCHS} epochs...\n")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} -> Loss: {train_loss/len(train_loader):.4f} | Accuracy: {train_acc:.2f}%")
        
    print("\nQuá trình huấn luyện hoàn tất.")
    
    # Đánh giá (Evaluation Loop)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
    # Hiển thị kết quả chuẩn Scikit-learn
    labels_present = np.unique(all_labels)
    target_names = [ID_TO_LABEL[idx] for idx in labels_present]
    
    print("\n--- BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    # Vẽ Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - CCNN 1D Heartbeat Classification")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Đã lưu ma trận nhầm lẫn vào file confusion_matrix.png")
    # plt.show() # Uncomment nếu muốn xem popup biểu đồ

if __name__ == "__main__":
    train_model()
