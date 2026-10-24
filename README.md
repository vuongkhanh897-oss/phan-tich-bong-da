# ⚽ ĐỒ ÁN KHOA HỌC DỮ LIỆU: HỆ THỐNG XỬ LÝ DỮ LIỆU LỚN, PHÂN TÍCH CHUỖI THỜI GIAN VÀ ĐỊNH GIÁ CẦU THỦ BẰNG MACHINE LEARNING

<div align="center">

| **Thông Tin Đồ Án** | **Chi Tiết** |
| :--- | :--- |
| **Tên Đề Tài** | Xây dựng Pipeline dữ liệu lớn, phân tích chuỗi thời gian (Panel Data) và dự đoán giá trị cầu thủ bóng đá |
| **Tác Giả** | Vương Đình Khánh |
| **Trường** | Đại học Phenikaa |
| **Công Nghệ Cốt Lõi** | Python, Streamlit, Pandas, Scikit-Learn, NumPy |
| **Nguồn Dữ Liệu** | EA Sports FC / FIFA Multi-Year Datasets (2020 – 2025) |

</div>

---

## 📖 1. TỔNG QUAN ĐỀ TÀI VÀ ĐẶT VẤN ĐỀ (INTRODUCTION & PROBLEM STATEMENT)

### 1.1. Bối cảnh nghiên cứu và bài toán thực tiễn
Thị trường chuyển nhượng bóng đá toàn cầu trong nhiều năm qua đã vươn lên thành một hệ sinh thái tài chính khổng lồ, nơi giá trị kinh tế của một cầu thủ không chỉ phản ánh qua năng lực kỹ thuật thuần túy trên sân cỏ mà còn chịu sự chi phối nặng nề từ các yếu tố động học phức tạp:
* **Chu kỳ sinh học và hệ số tuổi tác (`age`):** Sự suy giảm thể lực theo thời gian và mốc "đỉnh cao" sự nghiệp.
* **Giá trị tiềm năng kỳ vọng (`potential`):** Động lực đầu tư dài hạn của các câu lạc bộ lớn đối với các tài năng trẻ.
* **Biến động lạm phát tài chính qua các năm:** Sự trượt giá và thay đổi mặt bằng định giá chung của thị trường qua từng mùa giải.

### 1.2. Tính cấp thiết của việc ứng dụng Khoa học Dữ liệu
Việc định giá cầu thủ trong môi trường bóng đá truyền thống thường mang tính chủ quan cao dựa trên cảm tính của chuyên gia hoặc truyền thông. Đồ án này được xây dựng nhằm giải quyết bài toán cốt lõi: Thiết kế một hệ thống tự động hóa khép kín có khả năng tiếp nhận khối lượng dữ liệu lịch sử khổng lồ, làm sạch, chuẩn hóa chuỗi thời gian đa niên độ, từ đó khai thác tri thức và xây dựng mô hình trí tuệ nhân tạo để định giá cầu thủ một cách khách quan, định lượng và chính xác dựa trên bằng chứng dữ liệu thực tế.

---

## 📊 2. PHÂN TÍCH CHUYÊN SÂU NGUỒN DỮ LIỆU VÀ CƠ CHẾ NẠP (DATASET & INGESTION ANALYSIS)

### 2.1. Khảo sát cấu trúc nguồn dữ liệu thô (Dataset Origin)
Dự án khai thác bộ dữ liệu **EA Sports FC / FIFA Multi-Year Datasets** (do cộng đồng dữ liệu thể thao quốc tế và tác giả Stefano Leone cung cấp).
* **Phạm vi không gian và thời gian:** Dữ liệu bao phủ toàn bộ thông tin chi tiết của hàng vạn cầu thủ từ các phiên bản game **FIFA 20 đến EA Sports FC 25** (tương ứng chuỗi thời gian từ năm 2020 đến 2025).
* **Đặc thù phân mảnh:** Dữ liệu được lưu trữ phân rã trong các thư mục độc lập từ `archive (4)` đến `archive (10)`. Mỗi thư mục chứa các tệp `.csv` với hàng chục cột thuộc tính kỹ thuật, thể chất, thông tin cá nhân và giá trị định giá thị trường (`value_eur`) tại từng thời điểm phát hành.

### 2.2. Cơ chế thu thập tự động hóa thông minh (Auto Ingestion Engine)
Để giải quyết bài toán dữ liệu phân mảnh và sự sai lệch cấu trúc cột qua các năm, tập lệnh xử lý dữ liệu đã triển khai cơ chế tự động hóa:
* **Quét đệ quy hệ thống tệp tin (Recursive Scanning):** Thay vì nạp thủ công từng tệp, thuật toán tự động quét toàn bộ cây thư mục cục bộ chứa các gói `archive`.
* **Bóc tách định danh bằng biểu thức chính quy (Regex Versioning):** Tự động phân tích chuỗi đường dẫn và tên tệp để nhận diện, trích xuất chính xác năm phiên bản (`fifa_version`) gắn liền với từng dòng dữ liệu, chuyển đổi các tệp đơn lẻ thành một cấu trúc **Panel Data (Dữ liệu chuỗi thời gian)** đồng nhất trước khi bước vào giai đoạn ETL.

---

## 🏗️ 3. KIẾN TRÚC PIPELINE XỬ LÝ DỮ LIỆU LỚN (6-STAGE BIG DATA PIPELINE)

Hệ thống vận hành theo mô hình kỹ thuật dữ liệu (Data Engineering Pipeline) khép kín gồm 6 giai đoạn liên thông:

```text
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│ 1. Multi-Source │ ──> │ 2. Auto Ingestion    │ ──> │ 3. ETL Process  │
│ (Archive 4-10)  │     │ (Regex Versioning)   │     │ (Clean & Transform)
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                              │
┌─────────────────┐     ┌──────────────────────┐              │
│ 6. Visualization│ <── │ 5. ML Valuation      │ <────────────┘
│ (Streamlit Web) │     │ (Linear Regression)  │   4. Storage Layer
└─────────────────┘     └──────────────────────>  (Streamlit Cache)
Chi tiết các tầng xử lý kỹ thuật:
Stage 1 & 2 (Thu thập và Nhận diện): Tiếp nhận luồng dữ liệu thô lớn từ các thư mục lưu trữ và tự động gắn nhãn thời gian phiên bản.

Stage 3 (ETL - Trích xuất, Chuyển đổi và Làm sạch):

Feature Selection: Chỉ trích xuất các trường dữ liệu thực sự có ý nghĩa phân tích và định giá (age, overall, potential, value_eur, pace, shooting, passing, dribbling, defending, physic) nhằm tối ưu hóa tải trọng bộ nhớ RAM.

Schema Harmonization: Xử lý hiện tượng lệch tên cột giữa các phiên bản (ví dụ chuẩn hóa nationality thành nationality_name).

Missing Value Imputation: Tự động quét các giá trị khuyết thiếu (NaN) và lấp đầy bằng giá trị trung vị (median) tương ứng của từng thuộc tính để duy trì tính toàn vẹn của phân phối thống kê.

Stage 4 (Tầng lưu trữ đệm - Storage Layer): Ứng dụng cơ chế cache trong bộ nhớ (@st.cache_data). Dữ liệu sau khi qua ETL được lưu đệm trên RAM, giúp các thao tác truy vấn, lọc dữ liệu ở các phiên làm việc sau diễn ra gần như tức thời.

Stage 5 (Động cơ Học máy - ML Valuation Engine): Huấn luyện mô hình học có giám sát để thiết lập hệ số định giá tự động.

Stage 6 (Tầng trực quan hóa - Visualization): Truyền tải kết quả phân tích và định giá lên giao diện người dùng.

📸 Ảnh minh họa Giao diện Dashboard Tổng quan:

📈 4. PHÂN TÍCH KHÁM PHÁ DỮ LIỆU CHUYÊN SÂU (DEEP-DIVE EDA INSIGHTS)
Thông qua các công cụ truy vấn và phân tích đa chiều trên tập dữ liệu hợp nhất, đồ án đã khai thác được nhiều tri thức thực tiễn quan trọng:

4.1. Bản chất phi tuyến tính của mối quan hệ giữa Overall và Giá trị chuyển nhượng
Kết quả phân tích biểu đồ tương quan (Scatter Plot) chỉ ra rằng giá trị chuyển nhượng (value_eur) hoàn toàn không tăng trưởng theo đường thẳng tuyến tính đối với chỉ số tổng quát (overall).

Cụ thể, khi chỉ số tổng quát duy trì dưới ngưỡng 80, giá trị thị trường tăng trưởng rất chậm. Tuy nhiên, khi chạm và vượt qua mốc 85, đường cong giá trị bứt phá mạnh mẽ theo dạng hàm mũ (Exponential Growth). Điều này phản ánh chính xác quy luật kinh tế thực tế: sự khan hiếm nghiêm trọng của nhóm cầu thủ siêu sao đẳng cấp thế giới đẩy mức giá chuyển nhượng lên các kỷ lục phi tuyến tính.

4.2. Động học vòng đời sự nghiệp và hệ số tuổi tác (Age Lifecycle Dynamics)
Việc phân tích chuỗi thời gian qua các mùa giải minh chứng rằng đỉnh cao định giá thị trường của một cầu thủ tập trung trong khoảng độ tuổi vàng từ 25 đến 28 tuổi.

Đối với nhóm cầu thủ trên 30 tuổi, mặc dù năng lực chuyên môn (overall) có thể vẫn duy trì ở mức cao nhờ kinh nghiệm, hệ số tuổi tác bắt đầu tạo ra tác động âm rõ rệt vào mức định giá thị trường, phản ánh thực tế giảm giá trị tài sản dài hạn của các câu lạc bộ khi đầu tư vào cầu thủ lớn tuổi.

4.3. Vai trò quyết định của chỉ số Tiềm năng (potential) ở nhóm cầu thủ trẻ
Phân tích tương quan theo nhóm tuổi cho thấy đối với các cầu thủ dưới 22 tuổi, biến số potential đóng vai trò là trọng số chi phối mạnh mẽ nhất định hình giá trị thị trường, thể hiện bản chất "đầu tư vào tương lai" của thị trường chuyển nhượng hiện đại.

📸 Ảnh minh họa Biểu đồ Phân tích EDA:

🤖 5. PHƯƠNG PHÁP VÀ MÔ HÌNH HỌC MÁY (MACHINE LEARNING VALUATION METHODOLOGY)
5.1. Lựa chọn thuật toán và cơ chế học tập
Hệ thống sử dụng mô hình Hồi quy tuyến tính đa biến (Multiple Linear Regression) thuộc thư viện scikit-learn. Việc lựa chọn thuật toán này đảm bảo tính minh bạch và khả năng giải thích nhân quả cao, giúp phân tích tường minh mức độ tác động của từng chỉ số thành phần đối với giá trị bằng tiền mặt (EUR).

5.2. Kỹ thuật xây dựng đặc trưng (Feature Engineering)
Mô hình tiếp nhận tập đặc trưng đầu vào toàn diện gồm các biến định lượng phản ánh năng lực cốt lõi của cầu thủ:

Các thông tin định danh và thể chất: Tuổi (age), Chỉ số tổng quát (overall), Tiềm năng (potential).

Sáu chỉ số kỹ thuật chuyên môn thành phần: Tốc độ (pace), Sút (shooting), Chuyền (passing), Rê bóng (dribbling), Phòng thủ (defending), Thể lực (physic).

Thông qua quá trình huấn luyện trên dữ liệu lịch sử, thuật toán tự động học các trọng số tương quan để ánh xạ các thuộc tính năng lực này thành mức định giá thị trường khách quan.

5.3. Hậu xử lý kết quả dự đoán (Post-processing)
Để loại bỏ hoàn toàn các trường hợp dự đoán bất hợp lý (như giá trị âm hoặc giá trị quá thấp đối với cầu thủ chuyên nghiệp do đặc thù toán học của hồi quy), hệ thống ứng dụng bộ lọc chặn ngưỡng tối thiểu:

Python
predicted_value = np.clip(model.predict(X_test), a_min=200000, a_max=None)
Cơ chế này đảm bảo mọi cầu thủ trong hệ thống đều nhận được mức sàn định giá hợp lý tối thiểu là 200,000 EUR.

🖥️ 6. GIAO DIỆN VÀ PHÂN HỆ CHỨC NĂNG WEB APP (STREAMLIT OVERVIEW)
Hệ thống được đóng gói thành một ứng dụng Web trực quan, cho phép người dùng tương tác trực tiếp với cơ sở dữ liệu qua các phân hệ chính:

🔍 Tra cứu & Quỹ đạo sự nghiệp: Cho phép chọn cầu thủ và tra cứu thông tin linh hoạt theo từng năm phiên bản tùy ý (2020 – 2025) với các biểu đồ so sánh giá trị thực tế và giá trị dự đoán bởi AI.

📸 Ảnh minh họa:

⚖️ So sánh đối đầu (Head-to-Head): Đặt hai cầu thủ lên bàn cân so sánh song song toàn bộ các chỉ số chuyên môn và chênh lệch định giá theo từng năm cụ thể.

📸 Ảnh minh họa:

🏆 Top cầu thủ giá trị nhất: Bảng xếp hạng động hiển thị các ngôi sao đắt giá nhất hệ thống theo từng tiêu chí lựa chọn.

📈 Phân tích chuỗi thời gian (EDA): Trực quan hóa các biểu đồ phân tán tương quan đa chiều giữa chỉ số chuyên môn và giá trị thị trường.

⚙️ Trạng thái Pipeline & Dữ liệu: Giám sát thông số kỹ thuật hệ thống, tổng số lượng bản ghi sau ETL và kiểm tra cấu trúc dữ liệu thô.