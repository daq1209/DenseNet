# Kế hoạch thực hiện DoAnML - Kiến trúc DenseNet Custom

Mục tiêu là tái tạo lại trung thực kiến trúc thuật toán DenseNet theo đúng hướng dẫn trong `guild.md` bằng Python (PyTorch), thay thế cho mô hình thư viện có sẵn trong Google Colab, và nâng cấp mô hình với hàm kích hoạt Mish + SE Block.

## Danh sách công việc

### Component 1: Mã nguồn Mô hình (Model Source Code)

Tạo một file mới tên là `model.py` chứa cốt lõi thuật toán. File này sẽ chứa các block PyTorch sau:
1. **`SEBlock`**: Lớp xử lý Attention tích hợp Squeeze (`nn.AdaptiveAvgPool2d(1)`) và Excitation (chạy qua 2 lớp tuyến tính với Sigmoid) để scale lại các kênh đặc trưng. 
2. **`DenseLayer`**: Khối lớp đơn chứa Batch Normalization, hàm kích hoạt `nn.Mish`, và phép chập `nn.Conv2d(kernel_size=3)`.
3. **`DenseBlock`**: Khối lắp ghép chứa một vòng lặp các `DenseLayer`, sử dụng `torch.cat([x, new_features], 1)` ở đầu ra mỗi vòng lặp. Cài đặt chuẩn `out_channels = k = 12`.
4. **`TransitionLayer`**: Khối hạ chiều với Conv 1x1 và AvgPool 2x2.
5. **`DenseNetCustom`**: Kiến trúc tổng thể cấu hình L=40. Với L=40 thì số block là 3, mỗi block có 12 lớp (12x3 + 4 = 40). Chèn `SEBlock` vào ngay sau mỗi `DenseBlock`.

---

### Component 2: Môi trường Huấn luyện (Google Colab Workspace)

Chỉnh sửa bản phác thảo huấn luyện `DenseNet_Workspace.ipynb` để import mô hình tự chế:
1. **Thay thế thư viện torchvision**: Hiện tại notebook đang dùng `models.densenet121(num_classes=10)`. Gỡ bỏ dòng này, thay bằng lệnh import `from model import DenseNetCustom`.
2. **Khởi tạo mô hình**: Gọi `model = DenseNetCustom(growth_rate=12, block_config=(12, 12, 12), num_classes=10)` và đưa lên cấu hình GPU (`to(device)`).
3. **Số vòng huấn luyện (Epochs)**: Chỉnh tham số `num_epochs = 20` lên thành `num_epochs = 50` theo đúng chỉ đạo trong `guild.md`. Thêm block code sinh lệnh `%%writefile model.py` tại Colab để Colab có thể tải file này từ chính nội dung của Notebook.

## Kế hoạch Kiểm thử

### Kiểm thử Tự động
Bạn có thể tự tay chạy notebook này trực tiếp trên Google Colab. Mình sẽ tạo mã Python đảm bảo notebook có thể chạy 100% chuẩn xác mà không gặp lỗi syntax hay shape mismatch.

### Kiểm thử Thủ công
1. Đọc lướt qua file `model.py` được sinh ra xem đã có các class `DenseBlock`, `SEBlock`, `torch.cat` và `nn.Mish` chưa.
2. Kiểm tra log của Colab xem Loss có giảm qua 50 epochs không, để xác nhận đạo hàm chạy mượt.
