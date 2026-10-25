# ⚽ Hệ Thống Phân Tích Hiệu Suất & Định Giá Cầu Thủ Bóng Đá (Football Analytics & ML Valuation System)

> Đồ án chuyên sâu về Quy trình Xử lý Dữ liệu Lớn (Big Data Pipeline), Phân tích Thống kê và Xây dựng Mô hình Học máy Hồi quy dự đoán giá trị thị trường cầu thủ, tích hợp Web Dashboard tương tác.

---

## 📑 Mục Lục
1. [Giới Thiệu Bài Toán & Mục Tiêu](#1-giới-thiệu-bài-toán--mục-tiêu)
2. [Cấu Trúc Thư Mục Dự Án](#2-cấu-trúc-thư-mục-dự-án)
3. [Quy Trình Xử Lý Dữ Liệu Thô (ETL & Data Cleaning Pipeline)](#3-quy-trình-xử-lý-dữ-liệu-thô-etl--data-cleaning-pipeline)
4. [Trích Xuất & Lựa Chọn Đặc Trưng (Feature Engineering)](#4-trích-xuất--lựa-chọn-đặc-trưng-feature-engineering)
5. [Mô Hình Học Máy & Cơ Chế Hoạt Động (ML Valuation Engine)](#5-mô-hình-học-máy--cơ-chế-hoạt-động-ml-valuation-engine)
6. [Đánh Giá Hiệu Suất & Sai Số Mô Hình (Evaluation Metrics)](#6-đánh-giá-hiệu-suất--sai-số-mô-hình-evaluation-metrics)
7. [Trực Quan Hóa & Giao Diện Web Dashboard (Visualization)](#7-trực-quan-hóa--giao-diện-web-dashboard-visualization)
8. [Hướng Dẫn Cài Đặt & Chạy Chương Trình (Quick Start)](#8-hướng-dẫn-cài-đặt--chạy-chương-trình-quick-start)
9. [Kết Luận Và Hướng Phát Triển Trong Tương Lai](#9-kết-luận-và-hướng-phát-triển-trong-tương-lai)

---

## 1. Giới Thiệu Bài Toán & Mục Tiêu
* **Bối cảnh:** Thị trường chuyển nhượng bóng đá chịu ảnh hưởng bởi vô số biến số kinh tế và chuyên môn. Việc định giá cầu thủ bằng cảm tính hoặc qua các nguồn độc quyền thiếu đi tính minh bạch khoa học.
* **Mục tiêu:** Xây dựng một hệ thống khép kín bằng Python để:
  * Tự động hóa quá trình làm sạch và chuẩn hóa dữ liệu thô (`football_raw.csv`).
  * Khám phá các mối quan hệ phi tuyến tính giữa hiệu suất thi đấu và giá trị kinh tế.
  * Huấn luyện mô hình Hồi quy (Regression) dự đoán chính xác mức định giá cầu thủ (`value_eur`).
  * Cung cấp giao diện Web trực quan để tra cứu, lọc và so sánh trực tiếp các cầu thủ.

---

## 2. Cấu Trúc Thư Mục Dự Án
```text
phan-tich-bong-da/
│
├── images/                  # Thư mục chứa ảnh minh họa giao diện và biểu đồ
│   ├── anh.1.png
│   ├── anh.2.png
│   ├── anh.3.png
│   └── anh.4.png
│
├── .gitignore               # Cấu hình bỏ qua file rác hệ thống
├── dinh_gia_cau_thu.py      # Script chính xử lý dữ liệu, huấn luyện ML và chạy app
├── football_raw.csv         # Tập dữ liệu thô ban đầu
├── requirements.txt         # Danh sách thư viện phụ thuộc Python
└── README.md                # Tài liệu kỹ thuật dự án
3. Quy Trình Xử Lý Dữ Liệu Thô (ETL & Data Cleaning Pipeline)
Dữ liệu ban đầu (football_raw.csv) chứa nhiều nhiễu, giá trị khuyết và định dạng chưa chuẩn. Hệ thống thực hiện các bước xử lý sau trước khi đưa vào mô hình:
[ Dữ liệu thô: football_raw.csv ]
               │
               ▼
 1. Xử lý Missing Values ──> Lọc bỏ hoặc điền giá trị trung vị (Median) cho các cột thiếu thông tin.
               │
               ▼
 2. Chuẩn hóa định dạng ───> Chuyển đổi chuỗi tiền tệ (EUR, triệu/tỷ) thành kiểu số thực (float).
               │
               ▼
 3. Mã hóa biến phân loại ──> Chuyển đổi vị trí, quốc tịch sang dạng số nguyên/vector để máy hiểu.

4. Trích Xuất & Lựa Chọn Đặc Trưng (Feature Engineering)
Để tối ưu hóa độ chính xác của mô hình, hệ thống tiến hành trích xuất các nhóm đặc trưng trọng yếu:

Nhóm Nhân khẩu học: Độ tuổi (age) – yếu tố quyết định tiềm năng phát triển và biên độ giá trị tương lai của cầu thủ.

Nhóm Hiệu suất chuyên môn: Số bàn thắng (goals), số pha kiến tạo (assists), tổng số phút thi đấu, tỷ lệ chuyền bóng chính xác.

Nhóm Thị trường & Hợp đồng: Uy tín câu lạc bộ chủ quản, thời hạn hợp đồng còn lại.
5. Mô Hình Học Máy & Cơ Chế Hoạt Động (ML Valuation Engine)
Dưới đây là sơ đồ kiến trúc luồng vận hành chi tiết thể hiện cách mô hình học máy tiếp nhận dữ liệu đầu vào, xử lý toán học và trả về kết quả dự đoán:
CÁC ĐẶC TRƯNG ĐẦU VÀO (X):
  • Độ tuổi cầu thủ (Age)
  • Số bàn thắng & kiến tạo (Goals/Assists)
  • Hiệu suất thi đấu mỗi trận
  • Uy tín câu lạc bộ & Thời hạn hợp đồng
          │
          │  (Đưa vào ma trận tính toán)
          ▼
  ┌─────────────────────────────────────────────────┐
  │         MÔ HÌNH HỒI QUY (REGRESSION MODEL)      │
  │  • Thuật toán học máy tự động học quy luật      │
  │  • Điều chỉnh trọng số (Weights) của từng chỉ số │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  KẾT QUẢ ĐẦU RA (PREDICTION):
  • Giá trị chuyển nhượng dự đoán (Predicted Value)
          │
          │  (So sánh với giá trị thực tế y)
          ▼
  ĐÁNH GIÁ SAI SỐ (EVALUATION):
  • Đo lường mức độ chênh lệch qua các chỉ số MAE, RMSE, R²
Bản chất bài toán: Đây là bài toán Hồi quy có giám sát (Supervised Regression) do mục tiêu cần dự đoán là một biến số liên tục (value_eur).

Quá trình học của thuật toán: Dữ liệu được chia tập huấn luyện (80%) và kiểm thử (20%). Mô hình quét qua các đặc trưng đầu vào để tự động tối ưu hóa trọng số (weights), giảm thiểu sai số dự đoán so với thị trường thực tế.
6. Đánh Giá Hiệu Suất & Sai Số Mô Hình (Evaluation Metrics)MAE (Mean Absolute Error): Đo lường mức độ lệch tuyệt đối trung bình giữa giá dự đoán và giá thực tế.RMSE (Root Mean Squared Error): Kiểm soát tốt các trường hợp dự đoán sai lệch lớn nhờ căn bậc hai sai số bình phương trung bình.Hệ số xác định ($R^2 Score$): Phản ánh mức độ phần trăm mà các đặc trưng đầu vào giải thích được sự biến động giá trị cầu thủ.
7. Trực Quan Hóa & Giao Diện Web Dashboard (Visualization)
Kết quả phân tích sau khi qua bộ lọc mô hình được xuất ra tệp dữ liệu sạch và tích hợp trực tiếp lên giao diện Web tương tác:

[![Anh 1](https://raw.githubusercontent.com/vuongkhanh897-oss/phan-tich-bong-da/main/images/anh.1.png)](https://github.com/vuongkhanh897-oss/phan-tich-bong-da/blob/d391f9d54f37d06bf6775f211f4c381aa483332b/anh.1.png)
![Anh 2](https://raw.githubusercontent.com/vuongkhanh897-oss/phan-tich-bong-da/main/images/anh.2.png)
![Anh 3](https://raw.githubusercontent.com/vuongkhanh897-oss/phan-tich-bong-da/main/images/anh.3.png)
![Anh 4](https://raw.githubusercontent.com/vuongkhanh897-oss/phan-tich-bong-da/main/images/anh.4.png)
8. Kết Luận Và Hướng Phát Triển Trong Tương Lai
8.1. Kết luận
Đồ án đã xây dựng thành công một hệ thống xử lý dữ liệu và định giá cầu thủ tự động khép kín (End-to-End Pipeline), từ khâu thu thập, làm sạch dữ liệu thô cho đến việc huấn luyện mô hình học máy.

Mô hình Hồi quy đã thành công trong việc lượng hóa mối quan hệ giữa các chỉ số chuyên môn và giá trị kinh tế của cầu thủ, cung cấp cơ sở tham khảo minh bạch và khách quan cho giới chuyên môn.

Giao diện Web Dashboard trực quan giúp người dùng dễ dàng tra cứu, lọc dữ liệu và so sánh tương quan trực tiếp giữa các cầu thủ trên thị trường.

8.2. Hướng phát triển trong tương lai (Future Works)
Mở rộng nguồn dữ liệu thời gian thực (Real-time Data Integration): Tích hợp thêm API tự động cập nhật số liệu thống kê trận đấu hàng tuần thay vì chỉ dựa vào dữ liệu tĩnh (football_raw.csv).

Nâng cấp thuật toán học máy nâng cao: Thử nghiệm và so sánh hiệu suất giữa các mô hình học máy phức tạp hơn (như Random Forest Regressor, XGBoost, hoặc Neural Networks).

Bổ sung tính năng dự báo xu hướng tương lai: Xây dựng thêm mô hình dự đoán biên độ tăng/giảm giá trị của cầu thủ trong các mùa giải tiếp theo dựa trên chuỗi thời gian (Time-series Forecasting).

Phát triển ứng dụng Web hoàn chỉnh: Đưa hệ thống lên môi trường đám mây (Cloud Deployment như Render, AWS hoặc Vercel) để công khai ứng dụng cho cộng đồng người dùng truy cập.
