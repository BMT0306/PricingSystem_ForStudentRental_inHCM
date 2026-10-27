# ANTIGRAVITY / AI AGENT MASTER SPECIFICATION (`CLAUDE.md`)
## Project: HCMC Student Room Rental Price Prediction & Explainable AI System

> **HƯỚNG DẪN SỬ DỤNG CHO BẠN:** Hãy lưu file này với tên `ANTIGRAVITY_SPEC.md` (hoặc `CLAUDE.md`) ở thư mục gốc của project. Khi mở **Antigravity**, bạn chỉ cần ra lệnh: *"Đọc kỹ file `ANTIGRAVITY_SPEC.md` và thực thi tuần tự từng Phase theo đúng tất cả các ràng buộc kỹ thuật bên trong."*

---

## 1. SYSTEM ROLE & STRICT ENGINEERING CONSTRAINTS

You are an elite Senior Machine Learning Engineer executing an end-to-end Tabular Machine Learning project for a university final project targeting a 9.5+/10 grade. You MUST strictly obey all architectural, statistical, and domain constraints defined below.

### 1.1. Hard Domain Constraints (NON-NEGOTIABLE)
1. **Data Source:** Exclusively scrape from `https://phongtro123.com` focusing on Ho Chi Minh City (TP.HCM).
2. **Target Raw Dataset Size:** Must collect **at least $12,000$ raw rows** (`N_raw >= 12000`). If the general HCMC pagination does not yield $12,000$ rows, automatically iterate across all HCMC district URLs and sub-categories (`cho-thue-phong-tro`, `tim-nguoi-o-ghep`, `nha-cho-thue`, `cho-thue-can-ho-mini`, `cho-thue-mat-bang`) until `len(raw_df) >= 12000`.
3. **Area Filter (Strict):** Only retain listings where area in square meters satisfies:
   $$8.0 \le \text{area\_m2} \le 60.0$$
4. **Price Filter (Strict):** Convert all rental prices to Vietnamese Dong per month ($\text{VNĐ/tháng}$) as a clean `int64` / `float64` column `price_vnd`, and only retain listings satisfying:
   $$500,000 \le \text{price\_vnd} \le 10,000,000$$
5. **Geographic Filter (Strict):** Only retain listings located in the 22 districts/cities of Ho Chi Minh City (Quận 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, Bình Thạnh, Gò Vấp, Phú Nhuận, Tân Bình, Tân Phú, Bình Tân, TP. Thủ Đức [including old Q2, Q9, Thủ Đức], Bình Chánh, Nhà Bè, Hóc Môn, Củ Chi, Cần Giờ).
6. **Zero Data Leakage Rule:**
   - NEVER compute imputation statistics, scaling parameters (`StandardScaler`, `RobustScaler`), or target encoding mappings (`TargetEncoder`) on the full dataset before splitting.
   - Split data into `Train (70%)`, `Validation (15%)`, and `Test (15%)` stratified by `district` BEFORE any statistical transformation.
   - Seal the `Test` set completely until the final evaluation table.
7. **Reproducibility Rule:**
   - Pin `RANDOM_SEED = 42` across `numpy`, `random`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, and `optuna`.

---

## 2. REQUIRED REPOSITORY DIRECTORY STRUCTURE

Antigravity must create and maintain the exact directory structure below:

```text
hcm_room_price_ml/
├── ANTIGRAVITY_SPEC.md          # This master specification file
├── README.md                    # Documentation & execution guide
├── requirements.txt             # Pinned Python dependencies
├── config/
│   └── config.yaml              # Central hyperparameters, paths, and constraints
├── data/
│   ├── raw/
│   │   └── phongtro123_raw.csv  # Raw scraped dataset (>= 12,000 rows)
│   └── processed/
│       ├── cleaned_full.csv     # Cleaned dataset after domain & IQR filtering
│       ├── train.csv            # 70% training split
│       ├── val.csv              # 15% validation split
│       └── test.csv             # 15% hold-out test split
├── src/
│   ├── __init__.py
│   ├── crawler.py               # Multi-endpoint resilient scraper with checkpointing
│   ├── preprocessor.py          # Cleaning, regex NLP features, geospatial & IQR filter
│   ├── eda.py                   # Automated generation of 8 publication-ready charts
│   ├── trainer.py               # Pipeline training, Optuna tuning, 6-model comparison
│   └── explainer.py             # SHAP global/local explainability & error analysis
├── models/
│   └── v1/
│       ├── best_pipeline.joblib # Serialized full preprocessing + model pipeline
│       ├── metadata.json        # 8 MLOps artifacts (Chip Huyen Ch.10 standard)
│       └── leaderboard.csv      # Comparison metrics of all trained models
├── reports/
│   └── figures/                 # High-DPI PNG charts from EDA, Evaluation & SHAP
└── app/
    └── streamlit_app.py         # Interactive Room Valuation & SHAP Web App
```

---

## 3. STEP-BY-STEP MODULE SPECIFICATIONS FOR ANTIGRAVITY

### PHASE 1: DATA ACQUISITION (`src/crawler.py`)
- **Objective:** Scrape $\ge 12,000$ room rental listings in TP.HCM from `phongtro123.com`.
- **Implementation Requirements:**
  1. Use `requests` + `BeautifulSoup4` with rotating `User-Agent` headers, retry logic with exponential backoff, and polite rate limiting (`time.sleep(random.uniform(0.8, 1.8))`).
  2. **Multi-Source URL Queue:** Build a list of seed URLs covering:
     - General HCMC listings (`https://phongtro123.com/tinh-thanh/ho-chi-minh`)
     - Category-specific HCMC URLs (`cho-thue-phong-tro-ho-chi-minh`, `tim-nguoi-o-ghep-ho-chi-minh`, `cho-thue-can-ho-mini-ho-chi-minh`, `cho-thue-can-ho-dich-vu-ho-chi-minh`, `nha-cho-thue-ho-chi-minh`)
     - All 22 District-specific URLs in HCMC (e.g., `https://phongtro123.com/tinh-thanh/ho-chi-minh/quan-1`, `quan-5`, `quan-10`, `quan-binh-thanh`, `quan-go-vap`, `thanh-pho-thu-duc`, etc.).
  3. **Extracted Schema (`data/raw/phongtro123_raw.csv`):**
     - `post_id`: Unique identifier or hash of URL.
     - `url`: Listing URL.
     - `title`: Raw listing title.
     - `price_raw`: Raw price string (e.g., `"3.5 triệu/tháng"`).
     - `area_raw`: Raw area string (e.g., `"25m2"`).
     - `location_raw`: Raw district/city string (e.g., `"Quận 5, Hồ Chí Minh"`).
     - `description`: Summary/description text.
     - `crawled_at`: ISO timestamp.
  4. **Checkpointing:** Save incremental batches to `data/raw/phongtro123_raw.csv` every 10 pages so the scraper is fault-tolerant. Keep scraping across URLs until total rows $\ge 12,000$.

### PHASE 2: DATA CLEANING & FEATURE ENGINEERING (`src/preprocessor.py`)
- **Objective:** Transform `phongtro123_raw.csv` into a clean, feature-rich tabular dataset `cleaned_full.csv` and split into `train.csv`, `val.csv`, `test.csv`.
- **Implementation Requirements:**
  1. **Deduplication:** Drop duplicates by `url` and normalized `(title, price_raw, area_raw)`.
  2. **Price Parsing (`parse_price_vnd`):**
     - Convert `"triệu"`, `"tr"` $\rightarrow \times 1,000,000$ (handle decimals like `"3.5 triệu"`, `"3,5 triệu"`, `"3tr5"` $\rightarrow 3,500,000$).
     - Convert `"nghìn"`, `"ngàn"`, `"k"`, `"đ/tháng"` $\rightarrow \times 1,000$.
     - Drop rows with `"thỏa thuận"`, `"liên hệ"`, `"đ/ngày"`, or missing prices.
  3. **Area Parsing (`parse_area_m2`):**
     - Extract float value before `m2` or `m²`.
  4. **Hard Filtering Assertions:**
     ```python
     df = df[(df["area_m2"] >= 8.0) & (df["area_m2"] <= 60.0)]
     df = df[(df["price_vnd"] >= 500_000) & (df["price_vnd"] <= 10_000_000)]
     ```
  5. **District Standardization & Geospatial Coordinates:**
     - Map `location_raw` and `title`/`description` to canonical HCMC districts (`Quận 1`, `Quận 3`, ..., `TP. Thủ Đức`). Map old `"Quận 2"`, `"Quận 9"`, `"Quận Thủ Đức"` to `"TP. Thủ Đức"` (or keep sub-zone tags).
     - Compute `dist_to_center_km` (Haversine distance to Ben Thanh Market: $10.7725^\circ\text{N}, 106.6980^\circ\text{E}$) and `dist_to_sgu_km` (Haversine distance to Saigon University Campus 1, 273 An Duong Vuong, District 5: $10.7599^\circ\text{N}, 106.6822^\circ\text{E}$).
     - Assign `zone_group` (`Core_Center`, `Inner_City`, `Suburban`).
  6. **NLP Regex Binary Features (from `title + " " + description`):**
     - `has_ac`, `has_mezzanine`, `has_private_wc`, `has_kitchen`, `has_window_balcony`, `has_fridge_washer`, `no_curfew`, `is_shared_dorm`.
     - `amenity_score`: Sum of positive room amenities.
     - `text_length`: Word count of combined text.
  7. **Grouped IQR Outlier Filtering:**
     - Compute `price_per_m2 = price_vnd / area_m2`.
     - Within each `district`, remove rows outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ of `price_per_m2`.
  8. **Stratified Train/Val/Test Split:**
     - Split $70\%$ Train, $15\%$ Val, $15\%$ Test stratified by `district` with `random_state=42`.

### PHASE 3: AUTOMATED DEEP EDA (`src/eda.py`)
- Generate and save 7 high-resolution plots (`300 DPI`) into `reports/figures/`:
  1. `01_target_distribution_log_transform.png` (Raw `price_vnd` vs $\log_{10}(\text{price\_vnd})$ + Q-Q Plot).
  2. `02_price_by_district_boxplot.png` (Sorted horizontal boxplot of rental prices across 22 districts).
  3. `03_area_vs_price_scatter.png` (Area vs Price colored by `zone_group` with regression trendlines).
  4. `04_amenity_impact_bars.png` (Mean price comparison across the 8 extracted NLP binary amenities).
  5. `05_correlation_heatmap.png` (Pearson correlation matrix of numerical features).
  6. `06_spatial_distance_decay.png` (Price per $\text{m}^2$ vs `dist_to_center_km` and `dist_to_sgu_km`).
  7. `07_data_cleaning_funnel.png` (Bar chart of record counts at each filtering stage).
- Automatically generate a Markdown summary report `reports/eda_summary.md` with exact statistics (Mean, SD, Median, IQR, Skewness, Kurtosis).

### PHASE 4: MODEL TRAINING, OPTUNA TUNING & MLOPS STORE (`src/trainer.py`)
- **Target Transformation:** Wrap all regressors in `TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)` so training optimizes log-error while predictions output exact `VNĐ`.
- **Feature Preprocessing Pipeline:**
  - Numerical features (`area_m2`, `dist_to_center_km`, `dist_to_sgu_km`, `amenity_score`, `text_length` + binary flags): `SimpleImputer(strategy="median")` + `RobustScaler()`.
  - Categorical features (`district`, `zone_group`): `TargetEncoder(smooth="auto", cv=5)` (or native categorical handling for CatBoost).
- **Models to Benchmark (5-Fold CV on Train + Eval on Val & Test):**
  1. `DummyRegressor(strategy="median")` (Baseline)
  2. `Ridge(alpha=10.0)` ($L_2$ Linear Baseline)
  3. `Lasso(alpha=0.001)` ($L_1$ Linear Baseline)
  4. `RandomForestRegressor(n_estimators=300, random_state=42)`
  5. `XGBRegressor` (Tuned via Optuna, 30 trials)
  6. `LGBMRegressor` (Tuned via Optuna, 30 trials)
  7. `CatBoostRegressor` (Tuned via Optuna, 30 trials)
- **Evaluation Metrics (computed in real VNĐ):** `R2`, `RMSE`, `MAE`, `MAPE`, and `MedAE` (Median Absolute Error).
- **Artifact Saving (`models/v1/`):** Save `best_pipeline.joblib`, `leaderboard.csv`, and `metadata.json` following the 8-artifact Model Store schema.

### PHASE 5: EXPLAINABLE AI (`src/explainer.py`) & WEB APP (`app/streamlit_app.py`)
- **`src/explainer.py`:**
  - Compute SHAP values on `test.csv` using `shap.TreeExplainer` (or `shap.Explainer`).
  - Save `08_shap_summary_beeswarm.png`, `09_shap_feature_importance_bar.png`, `10_shap_waterfall_sample.png`, and `11_residual_diagnostics.png` (Predicted vs Actual + Residuals histogram).
- **`app/streamlit_app.py`:**
  - Clean, modern UI for SGU students: Select District, enter Area ($8 - 60\text{ m}^2$), toggle checkboxes for amenities (Máy lạnh, Gác lửng, WC riêng, Giờ tự do, Ở ghép...), input Landlord's asking price.
  - Display: Predicted Fair Price (VNĐ), Fair Price Range ($\pm \text{MAE}$), Deal Verdict (Hợp lý / Đắt / Rẻ), Distance to SGU & District 1, and an interactive SHAP Breakdown explaining how each feature added or subtracted money from the base price.

---

## 4. VERIFICATION CHECKLIST FOR ANTIGRAVITY
Before declaring any phase complete, run automated verification checks:
- [ ] `assert len(raw_df) >= 12000`, "Raw dataset must have at least 12,000 rows!"
- [ ] `assert clean_df['area_m2'].between(8.0, 60.0).all()`, "Area constraint violated!"
- [ ] `assert clean_df['price_vnd'].between(500_000, 10_000_000).all()`, "Price constraint violated!"
- [ ] `assert set(train_df.index).isdisjoint(set(test_df.index))`, "Data leakage detected between Train and Test!"
- [ ] Verify all plots exist in `reports/figures/` and `models/v1/best_pipeline.joblib` loads and predicts accurately.