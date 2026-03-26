import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

def apply_bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=5):
    """
    1. Apply Band-pass Filter (0.5 - 40 Hz)
    Đây là bước quan trọng nhất để loại bỏ cả nhiễu tần số thấp và cao cùng lúc.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_notch_filter(signal, fs, freq=50.0, q=30.0):
    """
    2. Remove Power Line Interference (50/60 Hz)
    Sử dụng bộ lọc Notch để cắt đúng tần số điện lưới. Ở một số nước (như Mỹ) là 60Hz, 
    ở VN/Châu Âu là 50Hz.
    """
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, q)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def remove_baseline_wander(signal, fs):
    """
    3. Remove Baseline Wander (Sự lang thang của đường nền do nhịp thở)
    Bằng cách dùng một bộ lọc High-pass Filter cực thấp (VD: 0.5 Hz) 
    hoặc dùng trung vị (median filter). Ở đây ta dùng High-pass 0.5 Hz.
    """
    nyquist = 0.5 * fs
    cutoff = 0.5 / nyquist
    b, a = butter(2, cutoff, btype='high')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def normalize_signal(signal):
    """
    4. Normalization (Chuẩn hóa biên độ)
    Biến đổi tín hiệu về khoảng [0, 1] hoặc chuẩn hóa Z-score (mean=0, std=1).
    Rất quan trọng cho Machine/Deep Learning để mạng hội tụ nhanh.
    Ở đây dùng Z-score normalization.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std > 0:
        normalized_signal = (signal - mean) / std
    else:
        normalized_signal = signal
    return normalized_signal


def process_and_plot(record_name='104', duration_sec=5):
    """ Tải dữ liệu, áp dụng toàn bộ Pipeline và vẽ biểu đồ so sánh """
    print(f"Loading record {record_name} (first {duration_sec}s)...")
    
    # Đọc bản ghi (từ mẫu 0 đến 5 giây)
    # Bản ghi 104 có nhiễu đường nền khá rõ
    record = wfdb.rdrecord(record_name, pn_dir='mitdb', sampfrom=0, sampto=360*duration_sec)
    fs = record.fs
    
    # Lấy kênh 1 (MLII)
    raw_signal = record.p_signal[:, 0]
    time = np.arange(len(raw_signal)) / fs
    
    # --- PIPELINE TIỀN XỬ LÝ ---
    # Ưu tiên thứ tự: Notch -> Bandpass (đã bao gồm chống nhiễu đường nền)
    
    # Bước 1: Notch 60Hz (Vì MIT-BIH thu thập ở Mỹ -> điện 60Hz)
    sig_notch = apply_notch_filter(raw_signal, fs, freq=60.0)
    
    # Bước 2 & 3: Bandpass 0.5 - 40 Hz (0.5Hz sẽ tự động khử Baseline Wander)
    sig_bandpass = apply_bandpass_filter(sig_notch, fs, lowcut=0.5, highcut=40.0)
    
    # Bước 4: Chuẩn hóa Normalization
    sig_normalized = normalize_signal(sig_bandpass)
    
    # --- TRỰC QUAN HÓA (PLOT CHUYÊN ĐỀ 3 BỘ LỌC + CHUẨN HÓA) ---
    plt.figure(figsize=(16, 12))
    
    # 1. Tín hiệu gốc
    plt.subplot(4, 1, 1)
    plt.plot(time, raw_signal, color='#d62728', linewidth=1.2)
    plt.title('1. Tín hiệu gốc (Raw ECG Signal)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude (mV)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Sau khi khử nhiễu điện lưới
    plt.subplot(4, 1, 2)
    plt.plot(time, sig_notch, color='#ff7f0e', linewidth=1.2)
    plt.title('2. Sau Lọc Notch 60 Hz (Power Line Interference Removed)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude (mV)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 3. Sau khi khử Band-pass (Baseline + Noise Cơ học)
    plt.subplot(4, 1, 3)
    plt.plot(time, sig_bandpass, color='#2ca02c', linewidth=1.5)
    plt.title('3. Sau Lọc Band-pass 0.5 - 40 Hz (Baseline Wander & Artifacts Removed)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude (mV)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 4. Sau khi Chuẩn hóa
    plt.subplot(4, 1, 4)
    plt.plot(time, sig_normalized, color='#1f77b4', linewidth=1.5)
    plt.title('4. Sau khi Chuẩn hóa Z-Score (Normalization)', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Ampl.', fontsize=12)
    plt.xlabel('Thời gian (giây)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # Hiển thị biểu đồ pop-up trực tiếp (thay vì lưu ra ảnh)
    plt.show()    
    print("=> Bảng điện tâm đồ trước/sau Baseline Wander đã được hiển thị trên màn hình!")

if __name__ == "__main__":
    process_and_plot('104', duration_sec=8) # Tăng lên 8s để thấy rõ chu kỳ thở nhấp nhô
