# 🏠 Hỗ Trợ Sinh Viên Tìm Trọ ở TP.HCM (Student Rental Price Evaluator & Ripoff Prevention)
> **Hệ thống Machine Learning thẩm định giá thị trường, phát hiện bị hét giá & bóc tách chi phí sinh hoạt cho sinh viên các trường Đại học tại TP.HCM**  
> **Kiến trúc:** Modular MLOps Pipeline theo chuẩn Chip Huyen (Chương 10)  
> **Phạm vi Địa lý:** 102 Phường/Xã mới thuộc TP.HCM (Nghị quyết 1685/NQ-UBTVQH15)

---

## 📌 1. Bức Tranh Tổng Quan & Kiến Trúc Dự Án

Dự án xây dựng một hệ thống Machine Learning hoàn chỉnh từ A đến Z phục vụ việc **thu thập dữ liệu địa phương, làm sạch dữ liệu với các ràng buộc nghiệp vụ chặt chẽ, khám phá dữ liệu chuyên sâu (EDA), huấn luyện mô hình dự đoán giá thuê phòng trọ (VNĐ) với ràng buộc đơn điệu toán học $\frac{\partial P}{\partial \text{Area}} \ge 0$, giải thích mô hình bằng SHAP (Explainable AI)**, kèm giao diện Web trực quan (Streamlit) cho sinh viên các trường Đại học tại TP.HCM.

### Sơ đồ Luồng Hoạt động (Pipeline Flow)
```
[phongtro123.com] 
       │
       ▼ (src/crawler.py: 2-layer parsing, checkpointing, Year >= 2026 filter)
[data/raw/phongtro123_raw.csv] (9,023 dòng thô)
       │
       ▼ (src/preprocessor.py: unit parsing, chuẩn hóa KTX, 102 phường mới, POI distances, grouped IQR)
[data/processed/cleaned_full.csv] (6,365 dòng sạch)
       │
       ├─► Stratified Split 70/15/15: train.csv (4,455), val.csv (955), test.csv (955)
       │
       ├─► (src/eda.py) 8 biểu đồ khoa học 300 DPI + reports/eda_summary.md
       │
       ├─► (src/trainer.py) Log-scale Pipeline, Monotone Constraints, 7 models, Quantile Bands Q10-Q90
       │        └─► models/v1/best_pipeline.joblib (LightGBM) & leaderboard.csv & metadata.json
       │
       ├─► (src/explainer.py) SHAP Global Beeswarm, Waterfall Local + Residual Diagnostics
       │
       └─► (app/streamlit_app.py) Web App: Radar Chống Hét Giá, Kịch Bản Đàm Phán, Bản Đồ Vệ Tinh Đa Lớp
```

---

## 🏆 2. Bảng Xếp Hạng Mô Hình (Benchmark Leaderboard)

| Xếp hạng | Thuật toán | Val MAE (VNĐ) | Test MAE (VNĐ) | Test MedAE (VNĐ) | Test $R^2$ | Đảm bảo đơn điệu diện tích |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| 🥇 **Hạng 1** | **LightGBM Regressor** | **662,998** | **672,783** | **500,718** | **0.7273** | ✅ **Tuyệt đối ($+1$)** |
| 🥈 **Hạng 2** | **XGBoost Regressor** | 677,467 | 671,867 | 518,707 | 0.7359 | ✅ **Tuyệt đối ($+1$)** |
| 🥉 **Hạng 3** | **Ridge Regression** | 711,718 | 727,406 | 571,378 | 0.6952 | ✅ Hệ số dương ($w > 0$) |
| 4 | Lasso Regression | 713,672 | 730,564 | 573,731 | 0.6933 | ✅ Hệ số dương ($w > 0$) |
| 5 | Random Forest | 738,864 | 727,821 | 560,519 | 0.6909 | ✅ `monotonic_cst` |
| 6 | CatBoost Regressor | 812,055 | 851,593 | 661,818 | 0.5623 | ✅ `monotone_constraints` |
| 7 | Dummy Baseline (Median) | 1,415,096 | 1,426,091 | 1,200,000 | -0.0120 | ❌ |

---

## 🚀 3. Hướng Dẫn Cài Đặt & Chạy Nhanh

### 3.1. Chạy 1-Click trên Windows
Nhấp đúp chuột vào file **`run_app.bat`** tại thư mục gốc của dự án. Ứng dụng sẽ tự động kích hoạt môi trường và mở trình duyệt web tại `http://localhost:8501`.

### 3.2. Chạy thủ công từ Terminal
```powershell
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy Web App Thẩm Định Giá
streamlit run app/streamlit_app.py
```

### 3.3. Tái lập toàn bộ Pipeline (End-to-End)
```powershell
# Thu thập dữ liệu
python src/crawler.py

# Làm sạch dữ liệu & trích xuất đặc trưng
python src/preprocessor.py

# Tự động xuất biểu đồ EDA
python src/eda.py

# Huấn luyện mô hình & cập nhật Leaderboard
python src/trainer.py

# Trích xuất biểu đồ giải thích SHAP
python src/explainer.py

# Kiểm tra bộ test ràng buộc đơn điệu toán học
python tests/test_monotonicity_and_constraints.py
```

---

## 📊 4. Danh Mục Đặc Trưng & Trụ Cột Đánh Giá

| Nhóm | Biến đặc trưng | Diễn giải |
|---|---|---|
| **Không gian** | `area_m2`, `room_type` | Diện tích thực tế, phân loại (phòng trọ, CH mini, KTX/Sleepbox, CHDV, nguyên căn) |
| **Vị trí & CBD** | `ward_new`, `dist_to_center_km`, `dist_to_sgu_km` | 102 phường mới, khoảng cách tới Chợ Bến Thành (Q.1) và các CBD |
| **Học tập** | `dist_nearest_uni_km`, `n_uni_within_2km` | Khoảng cách tới trường ĐH gần nhất, số trường ĐH trong bán kính 2km |
| **Y tế & An sinh** | `dist_nearest_hospital_km`, `n_hospitals_within_2km` | Khoảng cách tới BV lớn gần nhất, số bệnh viện trong bán kính 2km |
| **Giao thông** | `dist_nearest_metro_km` | Khoảng cách tới ga Metro Bến Thành - Suối Tiên gần nhất |
| **Tiện nghi** | `has_ac`, `has_mezzanine`, `has_private_wc`, `has_kitchen`, `has_balcony_window`, `has_fridge`, `has_washer`, `has_elevator`, `is_fully_furnished` | Máy lạnh, gác lửng, WC riêng, kệ bếp, ban công, tủ lạnh, máy giặt, thang máy, full nội thất |
| **Tự do & An ninh** | `no_curfew`, `not_shared_owner`, `has_security`, `has_parking_basement` | Giờ tự do, không chung chủ, bảo vệ/khóa vân tay, hầm để xe |
| **Chi phí ẩn** | `is_all_inclusive` | Giá trọn gói đã bao gồm điện, nước, wifi |
| **Tổng hợp** | `amenity_score` | Tổng điểm tiện nghi cơ bản ($0 - 13$) |

---

## ⚖️ 5. Tính Pháp Lý & Bản Quyền
- **Dữ liệu:** Thu thập từ nguồn thông tin công khai theo **Điều 25 Luật Sở hữu Trí tuệ** phục vụ học tập, nghiên cứu phi thương mại. Tuyệt đối tuân thủ **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân (không lưu trữ số CCCD, tài khoản, số điện thoại riêng của chủ nhà).
- **Mã nguồn & Bản đồ:** Sử dụng 100% thư viện mở (MIT, Apache 2.0, BSD) và dữ liệu bản đồ mã nguồn mở OpenStreetMap + Esri Imagery. Hoàn toàn không phát sinh chi phí thương mại (0 VNĐ).
