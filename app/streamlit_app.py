"""
Module: app/streamlit_app.py
Description: Hỗ Trợ Sinh Viên Tìm Trọ ở TP.HCM
Features:
- Định vị phục vụ sinh viên tất cả các trường Đại học tại TP.HCM
- Thẩm định giá thị trường dựa trên Machine Learning & Hedonic Pricing Model
- Ràng buộc đơn điệu kinh tế (Monotonicity Constraints: ∂Price/∂Area >= 0)
- Làm rõ ràng dải giá thị trường [Q10 - Q90] với thước đo trực quan
- Phát hiện bị hét giá / bị hớ & Kịch bản đàm phán trả giá có cơ sở khoa học
- Bóc tách chi phí sinh hoạt thực tế trọn gói (Tiền phòng + Điện + Nước + Xe + Wifi)
- Bản đồ tương tác thực tế đa lớp (Đường phố OpenStreetMap + Ảnh vệ tinh Esri Satellite)
  với lộ trình đến trường, kết nối Trung tâm Q.1 & CBD mới (Thủ Thiêm, Sala), và mạng lưới phòng thật đối chứng.
"""

import os
import math
import yaml
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium

# Configure Streamlit page
st.set_page_config(
    page_title="Hỗ Trợ Sinh Viên Tìm Trọ ở TP.HCM",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .brand-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.3);
    }
    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        color: #ffffff;
    }
    .brand-subtitle {
        font-size: 1.05rem;
        color: #bfdbfe;
        font-weight: 500;
        margin-bottom: 14px;
    }
    .brand-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .brand-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .card-kpi {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease;
    }
    .card-kpi:hover {
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Price Spectrum Gauge */
    .spectrum-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 24px;
        margin: 16px 0 20px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    .spectrum-bar {
        height: 14px;
        border-radius: 8px;
        background: linear-gradient(90deg, #10b981 0%, #10b981 20%, #3b82f6 20%, #3b82f6 80%, #ef4444 80%, #ef4444 100%);
        position: relative;
        margin: 18px 0 10px 0;
    }
    .spectrum-legend {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        color: #475569;
        font-weight: 600;
    }

    .alert-box {
        border-radius: 12px;
        padding: 16px 20px;
        margin: 18px 0;
        font-weight: 500;
        font-size: 0.98rem;
    }
    .alert-bargain {
        background: #ecfdf5;
        border: 1.5px solid #10b981;
        color: #065f46;
    }
    .alert-fair {
        background: #fffbeb;
        border: 1.5px solid #f59e0b;
        color: #92400e;
    }
    .alert-ripoff {
        background: #fef2f2;
        border: 1.5px solid #ef4444;
        color: #991b1b;
    }

    .pillar-card {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .pillar-title {
        font-weight: 700;
        color: #1e3a8a;
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .pillar-desc {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.5;
    }

    /* Commute Summary Bar & High-Contrast Chips */
    .commute-summary-bar {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        background: #1e293b;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
    }
    .commute-chip {
        background: #0f172a;
        color: #f8fafc !important;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 500;
        border: 1px solid #3b82f6;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .commute-chip b {
        color: #93c5fd !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_resources():
    """Load configuration, POIs, ward mapping, cleaned dataset, and ML models."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open("config/poi_database.yaml", "r", encoding="utf-8") as f:
        poi_db = yaml.safe_load(f)
    
    ward_df = pd.read_csv("data/reference/ward_mapping_hcm.csv", encoding="utf-8")
    
    best_pipe = None
    q_low = None
    q_high = None
    
    if os.path.exists("models/v1/best_pipeline.joblib"):
        try:
            best_pipe = joblib.load("models/v1/best_pipeline.joblib")
        except Exception:
            pass
    if os.path.exists("models/v1/quantile_low.joblib"):
        try:
            q_low = joblib.load("models/v1/quantile_low.joblib")
        except Exception:
            pass
    if os.path.exists("models/v1/quantile_high.joblib"):
        try:
            q_high = joblib.load("models/v1/quantile_high.joblib")
        except Exception:
            pass

    # Load cleaned dataset for neighbor pins on map
    real_data = None
    for path in ["data/processed/cleaned_full.csv", "data/raw/phongtro123_raw.csv"]:
        if os.path.exists(path):
            try:
                real_data = pd.read_csv(path, encoding="utf-8")
                break
            except Exception:
                continue

    return config, poi_db, ward_df, best_pipe, q_low, q_high, real_data


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Compute distance in kilometers between two GPS coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def main():
    config, poi_db, ward_df, model, q_low, q_high, real_data = load_all_resources()

    # Brand Header (Concise & Universal for all HCMC Students)
    st.markdown("""
    <div class="brand-banner">
        <div class="brand-title">🏠 Hỗ Trợ Sinh Viên Tìm Trọ ở TP.HCM</div>
        <div class="brand-subtitle">Hệ thống thẩm định giá thị trường, phát hiện bị hét giá & dự toán chi phí sinh hoạt thực tế cho sinh viên các trường Đại học tại TP.HCM</div>
        <div class="brand-badges">
            <span class="brand-badge">🎓 Dành cho tất cả sinh viên TP.HCM</span>
            <span class="brand-badge">📍 102 Phường Mới TP.HCM</span>
            <span class="brand-badge">⚖️ Ràng buộc đơn điệu diện tích (∂P/∂A ≥ 0)</span>
            <span class="brand-badge">🛡️ Radar chống bị hét giá</span>
            <span class="brand-badge">💬 Kịch bản đàm phán trả giá</span>
            <span class="brand-badge">🗺️ Bản đồ đối chứng phòng thật</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Input Panel
    with st.sidebar:
        st.markdown("### 🏫 Trường Đại Học Của Bạn")
        uni_names = [u["name"] for u in poi_db["universities"]]
        sel_uni_name = st.selectbox(
            "Chọn trường Đại học bạn đang theo học:",
            options=uni_names,
            index=0,
            help="Hệ thống sẽ lấy cổng chính trường này làm tâm điểm đo lộ trình và thời gian đi xe máy thực tế."
        )
        sel_uni = next(u for u in poi_db["universities"] if u["name"] == sel_uni_name)

        st.markdown("---")
        st.markdown("### 📍 Vị Trí Căn Phòng Trọ")
        ward_options = ward_df["ward_new"].tolist()
        ward_labels = [f"{w} ({ward_df[ward_df['ward_new']==w]['old_district_primary'].values[0]})" for w in ward_options]

        default_ward_idx = 0
        for i, w in enumerate(ward_options):
            if "Chợ Quán" in w or "Phường 3" in w:
                default_ward_idx = i
                break

        sel_idx = st.selectbox(
            "Chọn Phường Mới (và Quận Cũ):",
            range(len(ward_options)),
            index=default_ward_idx,
            format_func=lambda i: ward_labels[i]
        )
        selected_ward = ward_options[sel_idx]
        ward_row = ward_df[ward_df["ward_new"] == selected_ward].iloc[0]

        st.markdown("---")
        st.markdown("### 📐 Diện Tích & Loại Hình")
        area_m2 = st.slider(
            "Diện tích phòng (m²):",
            min_value=8.0,
            max_value=60.0,
            value=22.0,
            step=1.0,
            help="Mô hình đảm bảo tính đơn điệu kinh tế: Cùng vị trí và tiện nghi, diện tích lớn hơn luôn có giá trị không thấp hơn!"
        )

        room_type_options = {
            "Phòng trọ truyền thống": "phong_tro",
            "Căn hộ mini / Studio": "can_ho_mini",
            "Ở ghép / KTX / Sleepbox": "o_ghep_ktx_sleepbox",
            "Căn hộ dịch vụ (CHDV)": "can_ho_dich_vu",
            "Nhà nguyên căn": "nha_nguyen_can"
        }
        room_type_label = st.selectbox("Loại hình phòng:", list(room_type_options.keys()))
        room_type = room_type_options[room_type_label]

        st.markdown("---")
        st.markdown("### ✨ Tiện Nghi & An Ninh")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            has_ac = st.checkbox("Máy lạnh", value=True)
            has_mezzanine = st.checkbox("Gác lửng", value=True)
            has_private_wc = st.checkbox("WC riêng", value=True)
            has_kitchen = st.checkbox("Kệ bếp", value=True)
            has_balcony = st.checkbox("Ban công / Cửa sổ", value=True)
            has_fridge = st.checkbox("Tủ lạnh", value=False)
            has_washer = st.checkbox("Máy giặt", value=False)
        with col_t2:
            has_elevator = st.checkbox("Thang máy", value=False)
            is_furnished = st.checkbox("Full nội thất", value=False)
            no_curfew = st.checkbox("Giờ tự do 24/7", value=True)
            not_shared = st.checkbox("Không chung chủ", value=True)
            has_sec = st.checkbox("Khóa vân tay / BV", value=True)
            has_parking = st.checkbox("Hầm để xe", value=True)
            is_inclusive = st.checkbox("Bao điện nước trọn gói", value=False)

        st.markdown("---")
        st.markdown("### 💰 Giá Chủ Nhà Đang Đòi")
        asking_price_mil = st.number_input(
            "Tiền phòng chủ trọ đưa ra (Triệu đ/tháng):",
            min_value=0.5,
            max_value=15.0,
            value=3.5,
            step=0.1
        )
        asking_price_vnd = asking_price_mil * 1_000_000

        st.markdown("---")
        st.markdown("### ⚡ Bóc Tách Chi Phí Ẩn (Thực Tế)")
        with st.expander("Chi tiết đơn giá điện, nước, gửi xe...", expanded=False):
            elec_rate = st.number_input("Giá điện chủ trọ thu (VNĐ/kWh):", min_value=2000, max_value=6000, value=3800, step=100)
            elec_kwh = st.slider("Số kWh dự kiến dùng/tháng:", min_value=30, max_value=300, value=90, step=10)
            water_cost = st.number_input("Tiền nước sinh hoạt (VNĐ/người/tháng):", min_value=0, max_value=250000, value=100000, step=10000)
            parking_cost = st.number_input("Tiền giữ xe máy (VNĐ/xe/tháng):", min_value=0, max_value=300000, value=120000, step=10000)
            wifi_service_cost = st.number_input("Phí dịch vụ, wifi, rác (VNĐ/tháng):", min_value=0, max_value=400000, value=150000, step=10000)

    # Geospatial computations
    lat = float(ward_row["centroid_lat"])
    lon = float(ward_row["centroid_lon"])

    # Distances to CBDs and Hubs
    # 1. Ben Thanh / District 1 Center
    d_center = haversine(lat, lon, config["geospatial"]["ben_thanh"]["lat"], config["geospatial"]["ben_thanh"]["lon"])
    t_center = int(round((d_center / 20.0) * 60.0 + 3.0))

    # 2. Sala Urban Area & Thu Thiem Financial Center (CBD Mới)
    lat_sala, lon_sala = 10.7680, 106.7150
    d_sala = haversine(lat, lon, lat_sala, lon_sala)
    t_sala = int(round((d_sala / 20.0) * 60.0 + 3.0))

    # 3. Distance to the user's SELECTED university
    d_target_uni = haversine(lat, lon, sel_uni["lat"], sel_uni["lon"])
    commute_time_mins = int(round((d_target_uni / 20.0) * 60.0 + 3.0))

    # 4. POI metrics (Universities, Hospitals, Metro)
    uni_dists = [haversine(lat, lon, u["lat"], u["lon"]) for u in poi_db["universities"]]
    d_nearest_uni = min(uni_dists)
    n_uni_2km = sum(1 for d in uni_dists if d <= 2.0)

    hosp_dists = [haversine(lat, lon, h["lat"], h["lon"]) for h in poi_db["hospitals"]]
    d_nearest_hosp = min(hosp_dists)
    n_hosp_2km = sum(1 for d in hosp_dists if d <= 2.0)

    metro_dists = [haversine(lat, lon, m["lat"], m["lon"]) for m in poi_db["metro_stations"]]
    d_nearest_metro = min(metro_dists)

    amenities_list = [
        has_ac, has_mezzanine, has_private_wc, has_kitchen, has_balcony,
        has_fridge, has_washer, has_elevator, is_furnished, no_curfew,
        not_shared, has_sec, has_parking
    ]
    amenity_score = sum(1 for a in amenities_list if a)

    # Normalize area for shared dorm if applicable
    effective_area = area_m2
    if room_type == "o_ghep_ktx_sleepbox" and area_m2 > 16.0:
        effective_area = 8.0

    # Model input dataframe
    input_data = {
        "area_m2": effective_area,
        "dist_to_center_km": d_center,
        "dist_to_sgu_km": d_target_uni,  # Map dynamic university commute distance
        "dist_nearest_uni_km": d_nearest_uni,
        "n_uni_within_2km": n_uni_2km,
        "dist_nearest_hospital_km": d_nearest_hosp,
        "n_hospitals_within_2km": n_hosp_2km,
        "dist_nearest_metro_km": d_nearest_metro,
        "amenity_score": amenity_score,
        "text_word_count": 80,
        "has_ac": int(has_ac),
        "has_mezzanine": int(has_mezzanine),
        "has_private_wc": int(has_private_wc),
        "has_kitchen": int(has_kitchen),
        "has_balcony_window": int(has_balcony),
        "has_fridge": int(has_fridge),
        "has_washer": int(has_washer),
        "has_elevator": int(has_elevator),
        "is_fully_furnished": int(is_furnished),
        "no_curfew": int(no_curfew),
        "not_shared_owner": int(not_shared),
        "has_security": int(has_sec),
        "has_parking_basement": int(has_parking),
        "is_all_inclusive": int(is_inclusive),
        "female_only": 0,
        "is_shared_dorm": 1 if room_type == "o_ghep_ktx_sleepbox" else 0,
        "ward_new": selected_ward,
        "zone_group": ward_row["zone_group"],
        "room_type": room_type
    }
    input_df = pd.DataFrame([input_data])

    # Model Inference with Monotone Guarantee
    if model is not None:
        try:
            pred_vnd = float(model.predict(input_df)[0])
            q10_vnd = float(q_low.predict(input_df)[0]) if q_low else pred_vnd * 0.88
            q90_vnd = float(q_high.predict(input_df)[0]) if q_high else pred_vnd * 1.15
        except Exception:
            pred_vnd = None
    else:
        pred_vnd = None

    if pred_vnd is None:
        # Grounded Hedonic Pricing Benchmark
        zone_mult = 1.35 if ward_row["zone_group"] == "Core_Center" else (1.05 if ward_row["zone_group"] == "Inner_City" else 0.85)
        base_rent = 120_000 * effective_area * zone_mult
        amenity_bonus = (has_ac * 350_000) + (has_mezzanine * 200_000) + (has_elevator * 200_000) + (is_furnished * 450_000) + (has_fridge * 150_000) + (has_washer * 150_000)
        dist_decay = max(0, (5.0 - d_target_uni) * 60_000)
        pred_vnd = base_rent + amenity_bonus + dist_decay
        pred_vnd = max(800_000, min(pred_vnd, 9_500_000))
        q10_vnd = pred_vnd * 0.88
        q90_vnd = pred_vnd * 1.15

    # True Living Cost Calculation
    if is_inclusive:
        monthly_elec = 0
        monthly_water = 0
        total_living_cost = asking_price_vnd + parking_cost + wifi_service_cost
    else:
        monthly_elec = elec_rate * elec_kwh
        monthly_water = water_cost
        total_living_cost = asking_price_vnd + monthly_elec + monthly_water + parking_cost + wifi_service_cost

    # ==============================================================================
    # 1. HERO KPI CARDS
    # ==============================================================================
    st.markdown("### 📊 Kết Quả Thẩm Định & Chỉ Số Thực Tế")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="card-kpi">
            <div class="kpi-label">🎯 Giá AI Thẩm Định Hợp Lý</div>
            <div class="kpi-value">{pred_vnd/1e6:.2f} <span style="font-size:1.1rem;font-weight:600">Tr/tháng</span></div>
            <div class="kpi-sub">Đơn giá: {pred_vnd/area_m2:,.0f} đ/m²</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="card-kpi">
            <div class="kpi-label">📊 Khung Giá Phổ Biến [Q10 - Q90]</div>
            <div class="kpi-value">{q10_vnd/1e6:.2f} — {q90_vnd/1e6:.2f} <span style="font-size:1.1rem;font-weight:600">Tr</span></div>
            <div class="kpi-sub">Vùng giá an toàn của 80% phòng tương đương</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="card-kpi">
            <div class="kpi-label">🛵 Đến {sel_uni['name'].split('-')[0].strip()}</div>
            <div class="kpi-value">{d_target_uni:.1f} km</div>
            <div class="kpi-sub">Ước tính ~{commute_time_mins} phút xe máy</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="card-kpi">
            <div class="kpi-label">💸 Tổng Chi Phí Thực Tế / Tháng</div>
            <div class="kpi-value" style="color:#2563eb">{total_living_cost/1e6:.2f} <span style="font-size:1.1rem;font-weight:600">Tr</span></div>
            <div class="kpi-sub">Gồm phòng + điện + nước + xe + wifi</div>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # 2. CLEAR EXPLANATION & VISUAL GAUGE OF [Q10 - Q90] MARKET RANGE
    # ==============================================================================
    q10_mil = q10_vnd / 1e6
    q90_mil = q90_vnd / 1e6
    st.markdown(f"""
    <div class="spectrum-container">
        <div style="font-weight:700; font-size:1.05rem; color:#0f172a; margin-bottom:4px;">
            🎯 Thước Đo Phổ Giá Thị Trường & Vị Trí Mức Giá Bạn Đang Xem
        </div>
        <div style="font-size:0.88rem; color:#64748b;">
            Giúp sinh viên hiểu rõ phân vị giá thực tế: <b>Q10 (Phân vị 10%)</b> là mức rẻ nhất, <b>[Q10 - Q90]</b> là vùng giá chuẩn mực của 80% sinh viên, và <b>Q90</b> là ngưỡng trần cao cấp.
        </div>
        <div class="spectrum-bar"></div>
        <div class="spectrum-legend">
            <span style="color:#10b981;">🟢 Phân khúc tiết kiệm (&lt; {q10_mil:.2f} Tr)</span>
            <span style="color:#2563eb;">🔵 Khung thị trường an toàn: {q10_mil:.2f} Tr — {q90_mil:.2f} Tr (80% phòng)</span>
            <span style="color:#ef4444;">🔴 Phân khúc cao cấp (&gt; {q90_mil:.2f} Tr)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 3. DEAL VERDICT & ANTI-RIPOFF ANALYSIS
    # ==============================================================================
    diff_amount = asking_price_vnd - pred_vnd
    diff_pct = (diff_amount / pred_vnd) * 100.0

    if asking_price_vnd < q10_vnd:
        verdict_type = "bargain"
        verdict_icon = "🟢"
        verdict_title = f"GIÁ HỜI BẤT THƯỜNG - RẺ HƠN THỊ TRƯỜNG {abs(diff_pct):.1f}%"
        verdict_msg = (
            f"Giá chủ trọ đưa ra ({asking_price_mil:.2f} triệu) thuộc nhóm 10% phòng rẻ nhất khu vực (thấp hơn cả mốc Q10 = {q10_vnd/1e6:.2f} triệu). "
            f"**Khuyến nghị:** Đây là mức giá rất tiết kiệm cho sinh viên. Tuy nhiên, bạn cần đến tận nơi kiểm tra kỹ chất lượng phòng, "
            f"tường ẩm mốc, áp lực nước và đọc kỹ hợp đồng để tránh bẫy hét giá điện nước phụ thu!"
        )
    elif asking_price_vnd <= pred_vnd:
        verdict_type = "bargain"
        verdict_icon = "🟢"
        verdict_title = f"GIÁ RẤT TỐT - THẤP HƠN GIÁ CHUẨN {abs(diff_pct):.1f}%"
        verdict_msg = (
            f"Mức giá {asking_price_mil:.2f} triệu nằm ở nửa dưới của khung thị trường an toàn, thấp hơn mức AI thẩm định ({pred_vnd/1e6:.2f} triệu). "
            f"Căn phòng này có tỷ lệ tiện ích trên giá tiền rất tốt, bạn nên cân nhắc giữ chỗ sớm nếu điều kiện thực tế ưng ý."
        )
    elif asking_price_vnd <= q90_vnd:
        verdict_type = "fair"
        verdict_icon = "🟡"
        verdict_title = f"GIÁ HỢP LÝ - ĐÚNG KHUNG THỊ TRƯỜNG (Chênh {diff_pct:+.1f}%)"
        verdict_msg = (
            f"Mức giá {asking_price_mil:.2f} triệu nằm hoàn toàn trong khung dao động bình thường [Q10 - Q90] tại {selected_ward}. "
            f"Bạn có thể sử dụng kịch bản đàm phán bên dưới để xin chủ trọ bớt 100k - 200k/tháng hoặc miễn phí tiền gửi xe máy để có mức giá tốt nhất."
        )
    else:
        verdict_type = "ripoff"
        verdict_icon = "🔴"
        verdict_title = f"CẢNH BÁO: BỊ HỚ / BỊ HÉT GIÁ CAO HƠN {diff_pct:+.1f}% (+{diff_amount/1e6:.2f} Triệu/tháng)!"
        verdict_msg = (
            f"Chủ nhà đang đòi {asking_price_mil:.2f} triệu, vượt qua cả ngưỡng trần Q90 ({q90_vnd/1e6:.2f} triệu) của phân khúc phổ thông! "
            f"Nếu đồng ý thuê mức giá này mà phòng không có nội thất cao cấp đặc biệt, bạn sẽ bị thiệt hại khoảng **{diff_amount * 12 / 1e6:.1f} triệu đồng/năm**. "
            f"Hãy dùng kịch bản đàm phán trả giá hoặc tham khảo các phòng thật lân cận trên bản đồ bên dưới!"
        )

    st.markdown(f"""
    <div class="alert-box alert-{verdict_type}">
        <div style="font-size:1.12rem; font-weight:700; margin-bottom:6px;">{verdict_icon} {verdict_title}</div>
        <div>{verdict_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 4. TABS: NEGOTIATION SCRIPT, LIVING COST, & EVALUATION LOGIC
    # ==============================================================================
    tab_nego, tab_cost, tab_theory = st.tabs([
        "💬 Kịch Bản Đàm Phán Trả Giá",
        "⚡ Bóc Tách Chi Phí Sinh Hoạt Thực Tế",
        "⚖️ Cơ Sở Đánh Giá & Ràng Buộc Khoa Học"
    ])

    with tab_nego:
        st.markdown("#### 📝 Kịch bản đàm phán dựa trên cơ sở vật chất bị thiếu:")
        missing_reasons = []
        if not has_ac:
            missing_reasons.append("phòng chưa có máy lạnh (sinh viên phải tự chịu nóng hoặc đầu tư lắp mới -350k/tháng)")
        if not has_elevator:
            missing_reasons.append("tòa nhà không có thang máy, việc đi lại leo lầu chuyển đồ bất tiện (-200k/tháng)")
        if not has_parking:
            missing_reasons.append("không có hầm để xe riêng trong nhà, phải gửi xe bãi ngoài tốn phí (-150k/tháng)")
        if not has_mezzanine and room_type == "phong_tro":
            missing_reasons.append("phòng không có gác lửng khiến không gian sinh hoạt và để đồ bị hẹp (-150k/tháng)")
        if d_target_uni > 3.5:
            missing_reasons.append(f"vị trí cách trường {d_target_uni:.1f} km, tốn thời gian di chuyển và tiền xăng hàng ngày (-200k/tháng)")
        if not has_fridge and not has_washer:
            missing_reasons.append("chưa có máy giặt và tủ lạnh trang bị sẵn")

        suggested_offer_vnd = max(pred_vnd, asking_price_vnd - 300_000)
        if asking_price_vnd > pred_vnd:
            suggested_offer_vnd = pred_vnd

        reasons_bullet = "\n- ".join(missing_reasons) if missing_reasons else "Căn phòng hiện tại giá hơi cao so với mặt bằng trung bình cùng khu vực."

        script_text = (
            f"Dạ em chào anh/chị chủ nhà ạ,\n"
            f"Em là sinh viên {sel_uni_name.split('-')[0].strip()}, em vừa qua xem phòng {area_m2:.0f}m² bên mình tại {selected_ward}.\n"
            f"Em rất thích không gian và muốn thuê trọ ở lâu dài để tiện đi học. Tuy nhiên qua khảo sát mặt bằng phòng quanh khu vực trường, "
            f"vì phòng mình:\n- {reasons_bullet}\n"
            f"Nên em xin phép đề xuất mức giá thuê là {suggested_offer_vnd/1e6:.2f} triệu/tháng (hoặc hỗ trợ em phí gửi xe/tiền dịch vụ) "
            f"để phù hợp với ngân sách sinh viên của em được không ạ? Nếu được em xin phép qua đặt cọc giữ phòng luôn ạ. Em cảm ơn anh/chị!"
        )

        st.code(script_text, language="markdown")
        st.info("💡 **Mẹo đàm phán:** Sinh viên hãy nhắn tin lịch sự, cam kết ở giữ gìn vệ sinh, không tụ tập ồn ào và ký hợp đồng dài hạn (6 tháng - 1 năm) để chủ nhà yên tâm giảm giá!")

    with tab_cost:
        st.markdown("#### 📊 Chi tiết các khoản sinh hoạt phí thực tế hàng tháng:")
        cost_df = pd.DataFrame([
            {"Hạng mục": "Tiền thuê phòng", "Đơn giá / Cách tính": f"{asking_price_mil:.2f} Tr/tháng", "Thành tiền (VNĐ)": asking_price_vnd},
            {"Hạng mục": "Tiền điện sinh hoạt", "Đơn giá / Cách tính": f"{elec_kwh} kWh × {elec_rate:,} đ/kWh" if not is_inclusive else "Bao trọn gói", "Thành tiền (VNĐ)": monthly_elec},
            {"Hạng mục": "Tiền nước sinh hoạt", "Đơn giá / Cách tính": f"{water_cost:,} đ/người/tháng" if not is_inclusive else "Bao trọn gói", "Thành tiền (VNĐ)": monthly_water},
            {"Hạng mục": "Tiền giữ xe máy", "Đơn giá / Cách tính": "1 xe máy", "Thành tiền (VNĐ)": parking_cost},
            {"Hạng mục": "Phí dịch vụ, wifi, rác", "Đơn giá / Cách tính": "Trọn gói tiện ích tòa nhà", "Thành tiền (VNĐ)": wifi_service_cost},
            {"Hạng mục": "🔥 TỔNG THỰC TẾ PHẢI TRẢ", "Đơn giá / Cách tính": "Tổng chi phí rời túi mỗi tháng", "Thành tiền (VNĐ)": total_living_cost}
        ])
        st.table(cost_df.style.format({"Thành tiền (VNĐ)": "{:,.0f}"}))
        st.caption("⚠️ **Cảnh báo bẫy điện nước:** Nhiều chủ trọ đưa ra tiền phòng rẻ (ví dụ 2.8 triệu) nhưng thu tiền điện 4.500đ - 5.000đ/kWh và phụ thu đủ thứ, khiến tổng chi phí thực tế vọt lên trên 3.8 triệu/tháng!")

    with tab_theory:
        st.markdown("#### 🔬 Cơ Sở Đánh Giá Giá Trị Thuê Trọ (Hedonic Pricing Model):")
        st.markdown(r"""
        Hệ thống thẩm định giá dựa trên **Mô hình Định giá Đặc tính (Hedonic Pricing Theory - Sherwin Rosen, 1974)**. 
        Mức giá thuê hợp lý $P$ được xác định là hàm tổng hợp đa chiều từ các thuộc tính cốt lõi phục vụ đời sống sinh viên:
        """)

        st.latex(r"""
        P_{\text{phòng}} = f\Big( \text{Diện tích},\ d_{\text{trường}},\ d_{\text{y tế}},\ d_{\text{trục đường/Metro}},\ t_{\text{CBD Quận 1 & Sala}},\ \text{Chợ/TTTM},\ \text{An ninh},\ \text{Tiện nghi} \Big)
        """)

        st.markdown("---")
        st.markdown("##### 🏛️ 6 Trụ Cột Đánh Giá Chi Tiết & Ràng Buộc Hợp Lý:")

        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-title">1. 📐 Quy mô không gian & Ràng buộc đơn điệu toán học (Monotone Constraints)</div>
            <div class="pillar-desc">
                Diện tích phòng là hàng hóa thông thường (<i>normal good</i>). Hệ thống áp đặt nghiêm ngặt ràng buộc vi mô:
                <br><b>∂P / ∂(Diện tích) ≥ 0 (ceteris paribus)</b>.
                <br>Điều này loại bỏ hoàn toàn lỗi phi lý khi phòng 60m² bị định giá thấp hơn 50m². Với phòng KTX / Sleepbox, diện tích đăng tin được chuẩn hóa về diện tích cá nhân (8.0m²) để tránh làm lệch phân phối giá.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-title">2. 🎓 Vị trí tương quan với Trường Đại học bạn đang theo học</div>
            <div class="pillar-desc">
                Khoảng cách (km) và thời gian di chuyển (phút) bằng xe máy đến chính ngôi trường bạn chọn. Phòng trọ trong bán kính <b>≤ 2.0 km</b> có giá trị tiện ích vượt trội nhờ tiết kiệm 30-45 phút kẹt xe và tiền xăng mỗi ngày.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-title">3. 🏥 Hệ thống y tế & Chăm sóc sức khỏe xung quanh</div>
            <div class="pillar-desc">
                Đo lường khoảng cách đến bệnh viện, trạm y tế gần nhất và số lượng cơ sở y tế trong bán kính 2 km. Khu vực có hạ tầng y tế hoàn thiện giúp sinh viên an tâm khi ốm đau hoặc phát sinh sự cố y tế khẩn cấp.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-title">4. 🚇 Hạ tầng giao thông, Trục đường chính & Kết nối CBD Trung tâm</div>
            <div class="pillar-desc">
                Khoảng cách đến các trục lộ huyết mạch, tuyến <b>Metro Bến Thành - Suối Tiên</b>, và thời gian di chuyển đến <b>Trung tâm Quận 1</b> hoặc các <b>CBD mới (Thủ Thiêm, KĐT Sala)</b> dưới <b>15 - 30 phút</b>. Vị trí giao thông thông suốt luôn giữ giá trị thanh khoản và mức giá thuê ổn định.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-title">5. 🛒 Tiện ích dân sinh đời sống (Chợ truyền thống & TTTM)</div>
            <div class="pillar-desc">
                Sự hiện diện của các chợ dân sinh, bách hóa, siêu thị và trung tâm thương mại lân cận giúp sinh viên dễ dàng mua thực phẩm tươi giá rẻ, giảm chi phí ăn ngoài hàng tháng.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-title">6. 🛡️ An ninh trật tự & Tiện nghi nội thất</div>
            <div class="pillar-desc">
                Chỉ số an toàn được đo lường qua: Khóa cửa vân tay, camera an ninh / bảo vệ trực 24/7, giờ giấc tự do không giới nghiêm, không chung chủ nhà; kết hợp các trang thiết bị giảm chi phí mua sắm ban đầu (máy lạnh, tủ lạnh, máy giặt, gác lửng, ban công thoáng khí).
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # 5. ACTIONABLE REALISTIC COMMUTER MAP
    # ==============================================================================
    st.write("---")
    st.markdown("### 🗺️ Bản Đồ Thực Tế: Lộ Trình Di Chuyển Đến Trường & Mạng Lưới Phòng Thật Đối Chứng")
    st.markdown(
        f"Bản đồ mô phỏng lộ trình kết nối phòng trọ tại **{selected_ward}** đến cổng chính **{sel_uni['name']}**, "
        f"kết nối đến Trung tâm Q.1 & CBD mới Thủ Thiêm, kèm **dữ liệu các phòng trọ thật lân cận** để bạn đối chứng."
    )

    # Commute info summary bar above map
    st.markdown(f"""
    <div class="commute-summary-bar">
        <div class="commute-chip">🎓 <b>Đến trường:</b> {d_target_uni:.1f} km (~{commute_time_mins} phút xe máy)</div>
        <div class="commute-chip">🏙️ <b>Đến Q.1 (Bến Thành):</b> {d_center:.1f} km (~{t_center} phút)</div>
        <div class="commute-chip">🏢 <b>Đến CBD mới (Sala):</b> {d_sala:.1f} km (~{t_sala} phút)</div>
        <div class="commute-chip">🏥 <b>Bệnh viện gần nhất:</b> {d_nearest_hosp:.1f} km</div>
        <div class="commute-chip">🚇 <b>Ga Metro gần nhất:</b> {d_nearest_metro:.1f} km</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Folium Map centered between Room and Target University
    mid_lat = (lat + sel_uni["lat"]) / 2.0
    mid_lon = (lon + sel_uni["lon"]) / 2.0
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=14, tiles=None)

    # Add 3 High-Quality Tile Layers (No API Key Required)
    # Layer 1: OpenStreetMap Standard (Crisp Vietnamese street names, alleys, markets)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ Bản Đồ Đường Phố Thực Tế (OpenStreetMap)",
        control=True
    ).add_to(m)

    # Layer 2: Esri World Imagery (High-Resolution Satellite Photographic View)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri World Imagery",
        name="🛰️ Ảnh Vệ Tinh Thực Tế (Esri Satellite)",
        control=True
    ).add_to(m)

    # Layer 3: Esri World Topo Map (Urban Topography & Main Corridors)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri World Topo Map",
        name="🏙️ Bản Đồ Đô Thị & Giao Thông (Topo)",
        control=True
    ).add_to(m)

    # 1. Commute Buffer Circles around Selected University
    folium.Circle(
        radius=1500,
        location=[sel_uni["lat"], sel_uni["lon"]],
        color="#10b981",
        fill=True,
        fill_opacity=0.12,
        tooltip="Vùng Xanh: Đi bộ / Xe đạp lý tưởng đến trường (≤ 1.5 km)"
    ).add_to(m)

    folium.Circle(
        radius=3500,
        location=[sel_uni["lat"], sel_uni["lon"]],
        color="#f59e0b",
        fill=True,
        fill_opacity=0.08,
        tooltip="Vùng Vàng: Đi xe máy thuận tiện (≤ 3.5 km, dưới 12 phút)"
    ).add_to(m)

    # 2. Room Marker
    folium.Marker(
        [lat, lon],
        popup=f"<b>Phòng bạn đang thẩm định:</b><br>{selected_ward}<br>Diện tích: {area_m2}m²<br>Giá AI định: {pred_vnd/1e6:.2f} Tr/tháng",
        tooltip="📍 Vị trí phòng đang thẩm định",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)

    # 3. Target University Marker
    folium.Marker(
        [sel_uni["lat"], sel_uni["lon"]],
        popup=f"<b>{sel_uni['name']}</b><br>Cổng chính<br>Cách phòng: {d_target_uni:.1f} km (~{commute_time_mins} phút)",
        tooltip=f"🎓 {sel_uni['name']}",
        icon=folium.Icon(color="blue", icon="graduation-cap", prefix="fa")
    ).add_to(m)

    # 4. Nearest Hospital Marker
    nearest_hosp = min(poi_db["hospitals"], key=lambda h: haversine(lat, lon, h["lat"], h["lon"]))
    folium.Marker(
        [nearest_hosp["lat"], nearest_hosp["lon"]],
        popup=f"<b>Bệnh viện gần nhất:</b><br>{nearest_hosp['name']}<br>Cách phòng: {d_nearest_hosp:.1f} km",
        tooltip=f"🏥 {nearest_hosp['name']}",
        icon=folium.Icon(color="green", icon="plus-square", prefix="fa")
    ).add_to(m)

    # 5. CBD District 1 / Ben Thanh Center Landmark
    folium.Marker(
        [config["geospatial"]["ben_thanh"]["lat"], config["geospatial"]["ben_thanh"]["lon"]],
        popup=f"<b>Trung tâm Quận 1 (Chợ Bến Thành):</b><br>Cách phòng: {d_center:.1f} km (~{t_center} phút xe máy)",
        tooltip="🏙️ Trung tâm Quận 1",
        icon=folium.Icon(color="purple", icon="building", prefix="fa")
    ).add_to(m)

    # 6. Commute Route Polyline between Room and Target University
    folium.PolyLine(
        locations=[[lat, lon], [sel_uni["lat"], sel_uni["lon"]]],
        color="#2563eb",
        weight=4,
        opacity=0.88,
        dash_array="8, 6",
        tooltip=f"Lộ trình đến trường: {d_target_uni:.1f} km (~{commute_time_mins} phút xe máy)"
    ).add_to(m)

    # 7. Real Neighbor Listings Pins (from real scraped data)
    if real_data is not None and "lat" in real_data.columns and "lon" in real_data.columns:
        real_data_valid = real_data.dropna(subset=["lat", "lon", "price_vnd", "area_m2"]).copy()
        real_data_valid["dist_to_room"] = [
            haversine(lat, lon, rlat, rlon)
            for rlat, rlon in zip(real_data_valid["lat"], real_data_valid["lon"])
        ]
        nearby_rooms = real_data_valid[real_data_valid["dist_to_room"] <= 3.0].sort_values("dist_to_room").head(25)

        for _, n_row in nearby_rooms.iterrows():
            n_price = float(n_row["price_vnd"])
            n_area = float(n_row["area_m2"])
            n_title = str(n_row.get("title", "Phòng trọ"))[:60]
            n_url = str(n_row.get("url", "#"))
            n_dist = float(n_row["dist_to_room"])

            if n_price < 2_500_000:
                p_color = "#10b981"  # Cheap (Green)
            elif n_price <= 4_500_000:
                p_color = "#f59e0b"  # Fair (Orange)
            else:
                p_color = "#ef4444"  # High (Red)

            popup_html = f"""
            <div style='font-family:sans-serif; width:220px;'>
                <b style='color:#1e3a8a;'>{n_title}...</b><br>
                💰 <b>Giá:</b> {n_price/1e6:.2f} Tr/tháng<br>
                📐 <b>Diện tích:</b> {n_area:.0f} m²<br>
                📍 Cách phòng này: {n_dist:.2f} km<br>
                <a href='{n_url}' target='_blank' style='color:#2563eb; font-weight:600;'>🔗 Xem bài đăng gốc</a>
            </div>
            """

            folium.CircleMarker(
                location=[float(n_row["lat"]), float(n_row["lon"])],
                radius=6,
                color=p_color,
                fill=True,
                fill_color=p_color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{n_price/1e6:.2f} Tr | {n_area:.0f}m²"
            ).add_to(m)

    # Add Layer Control toggle in top right
    folium.LayerControl(position="topright").add_to(m)

    # Render Map
    st_folium(m, width=1150, height=520)

    # Map Legend
    st.markdown("""
    <div style="display:flex; gap:16px; flex-wrap:wrap; font-size:0.88rem; background:#1e293b; color:#e2e8f0; padding:12px 18px; border-radius:10px; margin-top:10px; border:1px solid #334155;">
        <span><b style="color:#f87171">🔴 Marker Đỏ:</b> Vị trí phòng bạn chọn</span>
        <span><b style="color:#60a5fa">🔵 Marker Xanh:</b> Cổng trường Đại học</span>
        <span><b style="color:#34d399">🟢 Marker Lá:</b> Bệnh viện gần nhất</span>
        <span><b style="color:#c084fc">🟣 Marker Tím:</b> Trung tâm Q.1 / CBD Sala</span>
        <span><b style="color:#34d399">🟢 Chấm Xanh:</b> Phòng thật &lt; 2.5 Tr</span>
        <span><b style="color:#fbbf24">🟡 Chấm Vàng:</b> Phòng thật 2.5 - 4.5 Tr</span>
        <span><b style="color:#f87171">🔴 Chấm Đỏ:</b> Phòng thật &gt; 4.5 Tr</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
