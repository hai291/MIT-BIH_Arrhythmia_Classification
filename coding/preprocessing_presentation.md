# Bài Thuyết Trình: Tiền Xử Lý Tín Hiệu Điện Tâm Đồ (ECG Preprocessing)
**Tiêu đề:** Tầm Quan Trọng Của Việc Tiền Xử Lý Đầu Ra Đối Với Mô Hình Học Sâu (Deep Learning) trong Phân Loại Nhịp Tim.

---

## Slide 1: Đặt Vấn Đề - "Garbage In, Garbage Out"
- **Tình trạng của tín hiệu ECG thực tế:** Dữ liệu thu thập từ các cảm biến thực tế (như bộ MIT-BIH) không bao giờ hoàn hảo. Chúng bị lẫn lộn bởi các yếu tố nhiễu vật lý và sinh lý.
- **Top 3 Kẻ Thù Của Tín Hiệu Sinh Học:**
  1. **Nhiễu điện lưới (Power-line Interference):** Tần số 50Hz/60Hz từ môi trường xung quanh lọt vào máy đo.
  2. **Nhiễu do nhịp thở (Baseline Wander):** Tần số siêu thấp (< 0.5Hz) làm đường nền tín hiệu bị trôi bấp bênh.
  3. **Nhiễu co cơ (EMG Artifacts):** Tần số cao (> 40Hz) do bệnh nhân vận động, ho, hoặc rùng mình.
- **Tác động nếu đưa "Tín hiệu thô" vào AI:** Mạng nơ-ron tích chập (CNN) sẽ bị phân tâm bởi các răng cưa nhiễu này thay vì tập trung học đặc trưng của bệnh lý (Hình thái học của QRS).

---

## Slide 2: Xóa Bỏ Nhiễu Điện Lưới với Notch Filter (60Hz)
- **Phương pháp thực hiện:** Áp dụng bộ lọc triệt tiêu dải băng (Notch Filter) tần số 60Hz.
- **Kết quả đầu ra:** Xóa sạch các gai nhiễu li ti, chằng chịt bám dọc theo đường sóng điện tim.
- **Tại sao mô hình lại cần điều này?** 
  - Các lớp *Convolutional Filters* (bộ lọc tích chập) của mạng CNN rất nhạy cảm với các đỉnh nhọn bất thường (high-frequency components). 
  - Việc loại bỏ răng cưa điện lưới giúp làm mịn đường sóng (Smooth curve), nhờ đó AI có thể "bắt" chính xác vị trí, độ hẹp/rộng và biên độ thực sự của phức bộ QRS thay vì bị cản trở bởi rác tín hiệu.

---

## Slide 3: Ổn Định Trục Tọa Độ với Bandpass Filter (0.5 - 40 Hz)
- **Phương pháp thực hiện:** 
  - Lọc dải cao qua (High-pass > 0.5Hz) để dẹp hiện tượng Baseline Wander.
  - Lọc dải thấp qua (Low-pass < 40Hz) để cạo đi các răng cưa co cơ (Artifacts).
- **Kết quả đầu ra:** Tín hiệu trải phẳng hoàn hảo trên cùng một đường nền trục hoành `y = 0`. Các nhiễu cơ bắp trên đỉnh R hoàn toàn biến mất.
- **Tại sao mô hình lại cần điều này?** 
  - **Chống Dương tính giả (False Positives):** Khi nhịp thở làm rớt đường nền chao đảo, một đỉnh nhịp bình thường có thể cắm ngược xuống dưới, khiến AI tưởng nhầm đó là nhịp Ngoại tâm thu thất dị dạng (V). Ép phẳng đường nền giúp cố định phương hướng của sóng P, Q, R, S, T.
  - **Giữ nguyên Hình dáng:** Việc cắt nhiễu < 40Hz vẫn bảo toàn 100% hình thái giải phẫu của tim, giúp CNN học được các mẫu "chuẩn mực" nhất.

---

## Slide 4: Chuẩn Hóa Z-Score (Normalization) - Chìa Khóa Của Thuật Toán Hội Tụ
- **Phương pháp thực hiện:** Chuẩn hóa toàn bộ ma trận dữ liệu về mức: `Trung bình (Mean) = 0`, `Độ lệch chuẩn (Std) = 1`.
- **Kết quả đầu ra:** Không còn khái niệm bệnh nhân tín hiệu "to" vì điện trở da thấp, bệnh nhân tín hiệu "nhỏ" vì máy đo rẻ tiền. Mọi bản ghi đều nằm gọn trong phạm vi biên độ chuẩn [-3, 3].
- **Tại sao mô hình RẤT cần điều này?**
  1. **Tránh Bùng nổ/Tiệt tiêu Gradient (Gradient Exploding/Vanishing):** Số liệu quá lớn hoặc quá chênh lệch sẽ khiến các hàm suy hao (Loss Function) trong mạng Neural Network đi vào vòng lặp vô tận, không thể hội tụ được trọng số.
  2. **Giải phóng không gian học (Feature Space):** AI không cần phải dành sức để suy luận xem "bản ghi này béo hay gầy". Z-Score buộc CNN phải gạt bỏ sự khác biệt về Amplitude (Biên độ), từ đó dồn 100% tài nguyên tính toán vào việc chẩn đoán hình dáng (Morphology) của dòng điện tim!

---

## Slide 5: Tổng Kết & Ý Nghĩa Lâm Sàng
- **Thông điệp cốt lõi:** Quá trình Tiền xử lý (Preprocessing) đã biến một chuỗi tín hiệu Vật Lý/Cơ Sinh Học đầy tạp âm và phi tuyến tính thành một **Vecto Toán Học Tinh Khiết**. 
- Nhờ sự "tinh khiết hóa" này, kiến trúc nội tại của mạng CNN 1D mới có thể phát huy sức mạnh trích xuất đặc trưng hình thái (Feature Extraction) đỉnh cao của nó, cải thiện vượt bậc **Độ Chính Xác (Accuracy)** và **F1-Score** trong bài toán dự đoán Rối Loạn Nhịp Tim!
