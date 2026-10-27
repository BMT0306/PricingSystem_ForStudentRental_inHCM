# BÁO CÁO TỔNG HỢP KHÁM PHÁ DỮ LIỆU (EDA SUMMARY REPORT)
## Đề tài: Hệ Thống Định Giá Phòng Trọ Sinh Viên Tại TP.HCM

---

### 1. Thống Kê Tổng Quan Bộ Dữ Liệu Sạch (`cleaned_full.csv`)
* **Tổng số bản ghi đạt chuẩn:** `6,365` phòng trọ.
* **Số lượng Phường mới phủ sóng:** `85` / 102 phường mới.
* **Số lượng Quận/Huyện cũ phủ sóng:** `21` khu vực.

---

### 2. Thống Kê Biến Mục Tiêu (`price_vnd`) & Biến Diện Tích (`area_m2`)

| Chỉ số Thống kê | Giá Thuê (VNĐ/tháng) | Giá Thuê (Triệu VNĐ) | Diện tích (m²) |
|---|---|---|---|
| **Trung bình (Mean)** | 4,271,390 | 4.27 | 26.6 |
| **Độ lệch chuẩn (Std)** | 1,712,314 | 1.71 | 9.9 |
| **Giá trị nhỏ nhất (Min)** | 500,000 | 0.50 | 8.0 |
| **Tứ phân vị 25% (Q1)** | 3,000,000 | 3.00 | 20.0 |
| **Trung vị (Median / Q2)** | 4,100,000 | 4.10 | 25.0 |
| **Tứ phân vị 75% (Q3)** | 5,500,000 | 5.50 | 32.0 |
| **Giá trị lớn nhất (Max)** | 10,000,000 | 10.00 | 60.0 |

---

### 3. Kiểm Định Phân Phối Mục Tiêu & Biến Đổi Logarit
* **Độ lệch phân phối gốc (Raw Skewness):** `0.404` $ightarrow$ Phân phối **lệch phải mạnh** (Right-skewed), tập trung nhiều phòng phân khúc $2.5 - 4.5$ triệu.
* **Độ lệch sau biến đổi $\ln(\text{price})$:** `-0.733` $\rightarrow$ Đạt tiệm cận chuẩn Gaussian ($|\text{Skew}| < 0.5$).
* **Kết luận phương pháp:** Áp dụng `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` cho toàn bộ các mô hình hồi quy.

---

### 4. Tỷ Lệ Xuất Hiện Các Tiện Ích Phòng Trọ Sinh Viên

| Tiện ích | Tỷ lệ xuất hiện (%) | Tác động giá trung bình |
|---|---|---|
| **Có máy lạnh (`has_ac`)** | 72.4% | Thường tăng từ +600k đến +1.2tr |
| **Có gác lửng (`has_mezzanine`)** | 35.9% | Phổ biến ở phân khúc sinh viên |
| **WC riêng (`has_private_wc`)** | 19.7% | Tiêu chí thiết yếu hàng đầu |
| **Kệ bếp (`has_kitchen`)** | 69.8% | Hỗ trợ sinh viên nấu ăn tiết kiệm |
| **Giờ tự do (`no_curfew`)** | 81.1% | Được ưa chuộng cao |
| **Bao điện nước (`is_all_inclusive`)** | 2.4% | Thường đi liền với mô hình KTX/Sleepbox |
| **Mô hình ở ghép/KTX (`is_shared_dorm`)** | 6.4% | Phân khúc giá rẻ $1.0 - 1.8$ triệu |
