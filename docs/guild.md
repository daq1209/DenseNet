# Hướng dẫn lập trình mạng DenseNet

Dưới đây là tóm tắt và giải thích chi tiết từng bước logic bạn cần lập trình.

## Tóm tắt 3 công việc cốt lõi của bạn
- **Xây dựng Kiến trúc Cơ sở (Baseline)**: Cài đặt mạng DenseNet gốc với cấu hình L=40 (tổng số lớp) và tốc độ tăng trưởng k=12.
- **Lập trình Cơ chế Ghép nối (Concatenation)**: Đảm bảo luồng dữ liệu (forward pass) sử dụng phép nối tensor thay vì phép cộng.
- **Thiết kế Phiên bản Nâng cấp (Upgraded Version)**: Nhúng khối Squeeze-and-Excitation (SE Block) và hàm kích hoạt Mish vào kiến trúc gốc.

## Giải thích chi tiết cách lập trình từng hạng mục

### 1. Xây dựng Khối Dày đặc (Dense Block)
Đây là nơi diễn ra thuật toán chính của bài báo. Thay vì tạo ra các lớp tuần tự, bạn sẽ tạo ra một lớp duy nhất tên là `DenseLayer` (chứa Batch Normalization, hàm kích hoạt, và Convolution 3x3).
- **Cách code**: Một `DenseBlock` sẽ chứa một danh sách (`ModuleList`) các `DenseLayer` này. Bạn dùng vòng lặp `for` để chạy qua từng lớp.
- **Logic Tốc độ tăng trưởng (k)**: Bạn phải thiết lập thông số `out_channels` của lớp Convolution đúng bằng k (ở đây k=12). Tức là dù đầu vào có dày đến đâu, lớp này chỉ xuất ra đúng 12 bản đồ đặc trưng mới (feature maps).

### 2. Xử lý phép Ghép nối Tensor (`torch.cat`)
Đây là điểm khác biệt chí mạng giữa DenseNet và ResNet. Bạn phải can thiệp vào hàm `forward(self, x)` của PyTorch.
- **Cách code**: 
  - Giả sử tensor đầu vào của Khối là `x`. Đi qua Lớp thứ nhất, sinh ra tensor mới là `new_features`.
  - Bạn **không** được lấy `x + new_features`. Bạn phải dùng lệnh: `x = torch.cat([x, new_features], 1)` (số 1 ở đây đại diện cho chiều của channel/kênh màu).
  - Tiếp tục đưa `x` (lúc này đã dày hơn) vào Lớp thứ hai, lặp lại quá trình nối này cho đến hết Khối.

### 3. Xây dựng Lớp Chuyển tiếp (Transition Layer)
Sau khi đi qua một `DenseBlock`, tensor của bạn đã bị nối thành một khối rất dày. Lớp này có nhiệm vụ "ép cân" giảm số lượng kênh.
- **Cách code**: Bạn định nghĩa một lớp gồm Convolution 1x1 để ép giảm số lượng kênh (channel), nối tiếp với một lớp Average Pooling 2x2 để giảm một nửa chiều dài và chiều rộng của bức ảnh.

### 4. Nâng cấp: Tích hợp hàm kích hoạt Mish
Bài báo gốc (2017) sử dụng ReLU. Bạn sẽ nâng cấp nó.
- **Cách code**: Rất đơn giản. Trong các hàm định nghĩa mạng, ở bất kỳ chỗ nào đang gọi `nn.ReLU(inplace=True)`, bạn chỉ cần thay thế bằng `nn.Mish(inplace=True)`. Điều này giúp đạo hàm không bị triệt tiêu khi đi qua các giá trị âm, mô hình sẽ học mượt hơn.

### 5. Nâng cấp: Nhúng Squeeze-and-Excitation (SE Block)
Đây là cơ chế Attention giúp mạng biết kênh dữ liệu nào là quan trọng để tập trung vào, kênh nào là nhiễu cần bỏ qua. Bạn sẽ code một class `SEBlock` riêng biệt và nhúng nó vào ngay sau mỗi `DenseBlock`.

**Cấu trúc SE Block bạn cần code**:
- **Squeeze (Ép)**: Dùng `nn.AdaptiveAvgPool2d(1)` để ép mỗi kênh (ví dụ kích thước 32x32) thành đúng 1 con số.
- **Excitation (Kích thích)**: Cho con số đó đi qua 2 lớp Linear (kết hợp với ReLU và Sigmoid) để tính ra một mảng "trọng số quan trọng" nằm trong khoảng từ 0 đến 1.
- **Scale (Nhân)**: Lấy mảng trọng số đó nhân ngược lại với tensor gốc. Kênh nào có trọng số gần 1 sẽ được giữ lại, kênh nào gần 0 sẽ bị làm mờ đi.

---

> **Kết luận:** Chỉ cần hoàn thành các module trên, bạn đã xây dựng thành công 100% "bộ não" của hệ thống, sẵn sàng để đồng đội gọi hàm và nạp dữ liệu vào huấn luyện.



## Định hướng 

Để tôi làm rõ sự khác biệt giữa quy mô của họ và quy mô nhóm bạn sẽ làm, để bạn hoàn toàn tự tin khi bảo vệ đồ án trước giảng viên:

### 1. Quy mô của bài báo gốc (Đẳng cấp Viện nghiên cứu)
- Nhóm tác giả đến từ Đại học Cornell, Đại học Thanh Hoa và Facebook AI Research. Họ có trong tay những dàn siêu máy tính với hàng tá GPU cực mạnh.
- Họ huấn luyện các mô hình khổng lồ lên tới 250 lớp (DenseNet-BC).
- Họ chạy trên tập dữ liệu ImageNet với 1,2 triệu bức ảnh độ phân giải cao và huấn luyện CIFAR lên tới 300 vòng (epochs).

### 2. Quy mô "DoAnML" của bạn (Chứng minh khái niệm - Proof of Concept)
- Mình làm ở quy mô **Tái tạo lõi thuật toán**. Nghĩa là bạn tự code lại đúng cái "linh hồn" của DenseNet: Cơ chế ghép nối (`torch.cat`) thay vì cộng dồn, và thuật toán tốc độ tăng trưởng (`k`).
- Nhóm bạn chỉ chạy một phiên bản "mini" (ví dụ: *DenseNet 40 lớp* ) trên tập dữ liệu nhỏ gọn CIFAR-10 (ảnh kích thước 32x32 pixel).
- Bạn chỉ cần chạy khoảng 30 - 50 epochs là đủ để báo cáo thành công rồi.