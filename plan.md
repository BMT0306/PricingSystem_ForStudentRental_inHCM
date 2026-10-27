# KẾ HOẠCH TỔNG THỂ ĐỒ ÁN MACHINE LEARNING TỪ A ĐẾN Z
## Đề tài: Hệ Thống Định Giá Thuê Phòng Trọ Cho Sinh Viên Tại TP.HCM (End-to-End ML Pipeline & Explainable AI)

---

## 1. BỨC TRANH TỔNG QUAN & LUỒNG DỮ LIỆU (ARCHITECTURE & DATA FLOW)

Để đạt điểm $9+$ (Điểm A) môn Machine Learning tại Đại học Sài Gòn (SGU), đồ án không được viết gộp lộn xộn trong một file duy nhất mà phải được thiết kế theo kiến trúc **Modular Pipeline** (tương tự tư duy MLOps ở Chương 10 sách Chip Huyen).

### Sơ đồ luồng hoạt động của hệ thống (System Pipeline)
1. **Tầng Thu Thập (`src/crawler.py`)** $\rightarrow$ Quét đa luồng danh mục & 22 Quận/Huyện TP.HCM trên `phongtro123.com` $\rightarrow$ Xuất bản `data/raw/phongtro_raw.csv` ($\ge 12,000$ dòng dữ liệu thô, có cơ chế lưu checkpoint chống mất dữ liệu).
2. **Tầng Tiền Xử Lý & Kỹ Thuật Đặc Trưng (`src/preprocessor.py`)** $\rightarrow$ Chuẩn hóa đơn vị VNĐ & $\text{m}^2$, lọc cứng theo ràng buộc ($8\text{ m}^2 \le \text{area} \le 60\text{ m}^2$ và $500,000\text{ VNĐ} \le \text{price} \le 10,000,000\text{ VNĐ}$), bóc tách 8 tiện ích bằng Regex NLP, tính khoảng cách địa lý Haversine, lọc ngoại lai theo IQR từng Quận $\rightarrow$ Xuất bản `data/processed/phongtro_cleaned.csv`.
3. **Tầng Khám Phá Dữ Liệu (`src/eda.py` & `notebooks/01_eda.ipynb`)** $\rightarrow$ Phân tích thống kê mô tả, kiểm định phân phối lệch phải (Right-skewed), biến đổi $\log_{10}(\text{price})$, xuất 8 biểu đồ phân tích chuyên sâu vào `reports/figures/`.
4. **Tầng Huấn Luyện & Tối Ưu Hóa (`src/trainer.py`)** $\rightarrow$ Chia tập dữ liệu `Train / Validation / Test` theo tỷ lệ $70\% / 15\% / 15\%$ có phân tầng theo Quận (`stratify=district`), đóng gói `Pipeline` chống rò rỉ dữ liệu (Data Leakage), huấn luyện 6 mô hình từ Baseline đến Ensemble, tối ưu siêu tham số bằng `Optuna` với 5-Fold Cross-Validation $\rightarrow$ Lưu mô hình tốt nhất vào `models/v1/`.
5. **Tầng Giải Thích Mô Hình & Ứng Dụng (`src/explainer.py` & `app/streamlit_app.py`)** $\rightarrow$ Phân tích đóng góp đặc trưng bằng **SHAP Values** và triển khai giao diện định giá trực tuyến cho sinh viên.

---

## 2. CHI TIẾT TỪNG GIAI ĐOẠN THỰC HIỆN (STEP-BY-STEP)

### GIAI ĐOẠN 1: THU THẬP DỮ LIỆU CHUYÊN SÂU (WEB SCRAPING 12,000+ DÒNG)
* **Vấn đề kỹ thuật:** Nếu chỉ cào 1 đường dẫn chung `https://phongtro123.com/tinh-thanh/ho-chi-minh?page=1..N`, khi vượt quá trang 400–500 web có thể giới hạn phân trang hoặc lặp lại tin VIP, không đủ $12,000$ dòng.
* **Giải pháp kỹ thuật trong đồ án:**
  1. **Chiến lược cào đa danh mục & đa Quận/Huyện (Multi-Endpoint Crawling):** Quét tuần tự qua trang tổng TP.HCM, kết hợp quét sâu vào từng danh mục con (`cho-thue-phong-tro`, `tim-nguoi-o-ghep`, `nha-cho-thue`, `cho-thue-can-ho-mini`) tại tất cả 22 Quận/Huyện & TP. Thủ Đức thuộc TP.HCM.
  2. **Cơ chế Checkpoint & Resume:** Cứ mỗi $10$ trang cào xong, hệ thống tự động ghi nối (append) xuống ổ đĩa `data/raw/phongtro_raw.csv`. Nếu rớt mạng hay tắt máy giữa chừng, khi chạy lại code sẽ đọc tiếp từ trang đang dở mà không mất dữ liệu cũ.
  3. **Bóc tách 2 lớp (Dual-Layer Parsing):** Kết hợp CSS Selectors chuẩn (`post-title`, `post-price`, `post-acreage`, `post-location`, `post-summary`) và bộ lọc **Regex Fallback** quét trực tiếp trên văn bản thẻ HTML để đảm bảo tỷ lệ lấy đủ trường đạt $>98\%$.

---

### GIAI ĐOẠN 2: LÀM SẠCH DỮ LIỆU & KỸ THUẬT ĐẶC TRƯNG (DATA CLEANING & FEATURE ENGINEERING)
Đây là giai đoạn quyết định $50\%$ số điểm của đồ án. Toàn bộ quy trình tuân thủ nghiêm ngặt 6 bước:

#### Bước 2.1: Khử trùng lặp (Deduplication)
* Loại bỏ các tin đăng trùng lặp do chủ trọ đăng lại nhiều lần dựa trên khóa phức hợp `(title_normalized, price_raw, area_raw, location_raw)` hoặc `post_url`.

#### Bước 2.2: Chuẩn hóa đơn vị Toán học (Unit Normalization)
* **Biến mục tiêu `price_vnd` (Đơn vị: VNĐ/tháng):**
  * Chuỗi `"3.5 triệu/tháng"`, `"3,5 tr"` $\rightarrow 3.5 \times 10^6 = 3,500,000\text{ VNĐ}$.
  * Chuỗi `"850 nghìn/tháng"`, `"850k"` $\rightarrow 850 \times 10^3 = 850,000\text{ VNĐ}$.
  * Chuỗi `"Thỏa thuận"`, `"Liên hệ"` hoặc tính theo `"đ/ngày"` $\rightarrow$ Loại bỏ hoàn toàn.
* **Biến diện tích `area_m2` (Đơn vị: $\text{m}^2$):**
  * Dùng biểu thức chính quy bắt số thực đứng trước `m2` hoặc `m²` (ví dụ: `"25.5 m2"` $\rightarrow 25.5$).

#### Bước 2.3: Áp dụng Ràng buộc miền giá trị (Hard Domain Constraints)
* Lọc giữ lại chính xác phân khúc phòng trọ/căn hộ mini sinh viên tại TP.HCM theo yêu cầu:
  $$8.0 \le \text{area\_m2} \le 60.0$$
  $$500,000 \le \text{price\_vnd} \le 10,000,000$$
* Loại bỏ các tin có địa chỉ ngoài TP.HCM (như Bình Dương, Đồng Nai, Long An bị đăng nhầm chuyên mục).

#### Bước 2.4: Trích xuất đặc trưng văn bản bằng NLP Regex (Text Feature Extraction)
Quét đồng thời trên `title` và `description` (đã đưa về chữ thường) để tạo $8$ đặc trưng nhị phân ($x_j \in \{0, 1\}$):
1. `has_ac`: Có máy lạnh / điều hòa.
2. `has_mezzanine`: Có gác lửng / gác cao / duplex.
3. `has_private_wc`: Có WC riêng / toilet riêng / khép kín.
4. `has_kitchen`: Có kệ bếp / bếp nấu ăn.
5. `has_window_balcony`: Có ban công / cửa sổ thoáng.
6. `has_fridge_washer`: Có tủ lạnh / máy giặt.
7. `no_curfew`: Giờ giấc tự do / không chung chủ / khóa vân tay.
8. `is_shared_dorm`: Phòng ở ghép / ký túc xá (KTX) / sleepbox (Đặc trưng cực kỳ quan trọng để giải thích các mức giá từ $500,000\text{ VNĐ}$ đến $1,500,000\text{ VNĐ}$).
* Tổng hợp thêm biến `amenity_score` (Điểm tiện ích tổng hợp):
  $$\text{amenity\_score} = \sum_{k=1}^{7} \text{feature}_k$$

#### Bước 2.5: Kỹ thuật Đặc trưng Địa lý (Geospatial Feature Engineering)
* Chuẩn hóa tên $22$ Quận/Huyện/TP. Thủ Đức tại TP.HCM.
* Ánh xạ tọa độ tâm $(\text{lat}, \text{lon})$ của từng Quận/Huyện và áp dụng **công thức Haversine** tính khoảng cách đường chim bay (km):
  $$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
  * `dist_to_center_km`: Khoảng cách từ phòng trọ đến trung tâm Quận 1 (Chợ Bến Thành).
  * `dist_to_sgu_km`: Khoảng cách từ phòng trọ đến cơ sở chính Đại học Sài Gòn (273 An Dương Vương, Quận 5).
* Phân nhóm khu vực (`zone_group`): Khu Trung tâm (Q1, Q3, Q4, Q5, Q10, Phú Nhuận, Bình Thạnh), Khu Nội thành mở rộng (Q6, Q7, Q8, Q11, Tân Bình, Tân Phú, Gò Vấp), Khu Ngoại thành & Đô thị mới (TP. Thủ Đức, Q12, Bình Tân, Bình Chánh, Nhà Bè, Hóc Môn).

#### Bước 2.6: Lọc ngoại lai theo từng Quận (Grouped IQR Outlier Removal)
* Một phòng $15\text{ m}^2$ giá $9,500,000\text{ VNĐ}$ ở Quận 1 có thể hợp lý (căn hộ dịch vụ trung tâm), nhưng ở Hóc Môn là nhiễu (nhập dư số 0).
* Do đó, áp dụng lọc ngoại lai trên biến `price_per_m2` ($\text{VNĐ/m}^2$) **theo từng nhóm Quận (`groupby('district')`)**:
  $$\text{IQR}_d = Q_{3,d} - Q_{1,d}$$
  $$\text{Hợp lệ khi: } Q_{1,d} - 1.5 \times \text{IQR}_d \le \text{price\_per\_m2} \le Q_{3,d} + 1.5 \times \text{IQR}_d$$

---

### GIAI ĐOẠN 3: KHÁM PHÁ DỮ LIỆU CHUYÊN SÂU (EXPLORATORY DATA ANALYSIS - EDA)
Nhóm sẽ xuất ra **8 biểu đồ khoa học** (lưu độ phân giải cao $300\text{ DPI}$) để đưa vào báo cáo và slide:
1. **Biểu đồ 1 (Target Distribution & Log Transformation):** So sánh phân phối của `price_vnd` gốc (lệch phải, độ lệch $\text{Skewness} > 1$) và sau khi lấy $\log(\text{price\_vnd})$ (tiệm cận phân phối chuẩn Gaussian). Kèm kiểm định thống kê.
2. **Biểu đồ 2 (Boxplot Price by District):** Phân bố giá thuê phòng trọ qua 22 Quận/Huyện được sắp xếp theo trung vị giảm dần.
3. **Biểu đồ 3 (Scatter Plot & Regression Line):** Tương quan giữa Diện tích (`area_m2`) và Giá thuê (`price_vnd`), phân màu theo `zone_group`.
4. **Biểu đồ 4 (Amenity Impact Barplot):** Mức chênh lệch giá trung bình khi phòng có vs. không có các tiện ích (`has_ac`, `has_mezzanine`, `has_private_wc`, `is_shared_dorm`).
5. **Biểu đồ 5 (Correlation Heatmap):** Ma trận tương quan Pearson & Spearman giữa toàn bộ các biến số học.
6. **Biểu đồ 6 (Distance Decay Curve):** Sự suy giảm của đơn giá thuê trên mỗi $\text{m}^2$ khi khoảng cách tới Quận 1 (`dist_to_center_km`) và ĐH Sài Gòn (`dist_to_sgu_km`) tăng lên.
7. **Biểu đồ 7 (Data Funnel Chart):** Thống kê số lượng mẫu qua từng phễu lọc (Raw $\rightarrow$ Deduplicated $\rightarrow$ Valid Range $\rightarrow$ IQR Cleaned).
8. **Biểu đồ 8 (Interactive Map - Folium):** Bản đồ nhiệt giá phòng trọ toàn TP.HCM.

---

### GIAI ĐOẠN 4: HUẤN LUYỆN MÔ HÌNH, CHỐNG DATA LEAKAGE & TỐI ƯU SIÊU THAM SỐ

#### 1. Nguyên tắc bất khả xâm phạm (Zero Data Leakage)
* Chia dữ liệu thành `Train (70%)`, `Validation (15%)`, `Test (15%)` **TRƯỚC KHI** thực hiện bất kỳ phép tính thống kê nào (`StandardScaler`, `TargetEncoder`, `SimpleImputer`).
* Mọi phép biến đổi đặc trưng bắt buộc đóng gói trong `sklearn.pipeline.Pipeline` và `ColumnTransformer` (chỉ gọi `.fit()` trên `X_train`, chỉ gọi `.transform()` trên `X_val` và `X_test`).

#### 2. Biến đổi hàm mục tiêu (Log-Target Transformation)
* Huấn luyện mô hình trên biến mục tiêu $y_{\text{train}}' = \ln(y_{\text{train}})$.
* Khi dự đoán, biến đổi ngược về đơn vị VNĐ bằng hàm mũ $\hat{y} = \exp(\hat{y}')$ thông qua `TransformedTargetRegressor` của `scikit-learn`.

#### 3. Hệ thống 6 Mô hình Thực nghiệm (Từ Baseline đến SOTA Tabular)
1. **Baseline 1 - Dummy Regressor:** Dự đoán bằng giá trị trung vị (Median) theo từng Quận.
2. **Baseline 2 - Linear Regression & Ridge/Lasso ($L_1, L_2$ Regularization):** Kiểm tra giả định tuyến tính.
3. **Non-linear Classical - K-Nearest Neighbors (KNN Regressor):** Định giá dựa trên $K$ căn phòng tương đồng nhất.
4. **Ensemble Bagging - Random Forest Regressor:** Giảm phương sai (Variance) bằng tập hợp cây quyết định.
5. **Ensemble Boosting 1 - XGBoost & LightGBM Regressor:** Tối ưu hóa Gradient Boosting tốc độ cao.
6. **Ensemble Boosting 2 (Trọng tâm) - CatBoost Regressor:** Xử lý trực tiếp biến phân loại (`district`, `ward`, `zone_group`) bằng cơ chế Ordered Target Statistics giúp chống Overfitting vượt trội.

#### 4. Hệ thống Metric Đánh giá Toàn diện
Không chỉ dùng một chỉ số, báo cáo sẽ đối sánh đồng thời 4 chỉ số trên đơn vị tiền tệ thực (VNĐ):
* **$R^2\text{ Score}$ (Hệ số xác định):**
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
* **$\text{RMSE}$ (Root Mean Squared Error - Phạt nặng các sai số lớn):**
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
* **$\text{MAE}$ (Mean Absolute Error - Sai số tuyệt đối trung bình tính bằng VNĐ):**
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
* **$\text{MAPE}$ (Mean Absolute Percentage Error - Tỷ lệ phần trăm sai số):**
  $$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

---

### GIAI ĐOẠN 5: GIẢI THÍCH MÔ HÌNH (XAI VỚI SHAP) & TRIỂN KHAI WEB APP (STREAMLIT)
1. **SHAP (Shapley Additive exPlanations):**
   * **Global Interpretability (SHAP Summary Beeswarm Plot):** Chứng minh trên toàn tập dữ liệu TP.HCM, yếu tố nào quyết định giá mạnh nhất (ví dụ: `area_m2` $\rightarrow$ `dist_to_center_km` $\rightarrow$ `is_shared_dorm` $\rightarrow$ `has_ac`).
   * **Local Interpretability (SHAP Waterfall Plot):** Giải thích cho từng căn phòng cụ thể (ví dụ: Giá nền $3.5\text{ triệu}$ $+ 800\text{k}$ do ở Quận 5 $+ 400\text{k}$ do có máy lạnh $- 300\text{k}$ do diện tích nhỏ $15\text{ m}^2 = 4.4\text{ triệu VNĐ}$).
2. **Streamlit Web App (`app/streamlit_app.py`):**
   * Cho phép sinh viên nhập thông tin phòng trọ đang định thuê $\rightarrow$ Hệ thống trả về **Mức giá hợp lý**, **Khoảng tin cậy dự đoán**, **Đánh giá Đắt/Rẻ/Hợp lý** so với giá chủ nhà đòi, kèm **Biểu đồ SHAP giải thích trực tiếp**.