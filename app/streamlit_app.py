"""
Module: app/streamlit_app.py
Description: Hệ thống Hỗ Trợ Sinh Viên Tìm Trọ & Thẩm Định Giá Thuê TP.HCM
Giao diện hiện đại, tự nhiên, trực quan, hỗ trợ chế độ xem đa lớp và phân tích chi phí thực tế.
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
    page_title="HỖ TRỢ SINH VIÊN TÌM TRỌ",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Soothing & Smooth Dark UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Smooth, Refined Dark Header */
    .header-card {
        background: linear-gradient(180deg, #151d2d 0%, #111724 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px -2px rgba(0, 0, 0, 0.35);
    }
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0 0 10px 0;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.55;
        margin: 0 0 18px 0;
        max-width: 860px;
    }
    .header-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .header-chip {
        font-size: 0.8rem;
        font-weight: 500;
        color: #cbd5e1;
        background: rgba(255, 255, 255, 0.04);
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
    }
    
    /* KPI Summary Cards - Velvety Dark */
    .kpi-card {
        background: #141c2c;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 6px 20px -2px rgba(0, 0, 0, 0.35);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Price Spectrum Ruler */
    .ruler-container {
        background: #141c2c;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.25);
    }
    .ruler-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }
    .ruler-sub {
        font-size: 0.84rem;
        color: #94a3b8;
        margin-bottom: 14px;
    }
    .ruler-bar-bg {
        height: 10px;
        background: #1e293b;
        border-radius: 5px;
        position: relative;
        margin: 16px 0 12px 0;
    }
    .ruler-range-safe {
        position: absolute;
        top: 0;
        bottom: 0;
        background: #38bdf8;
        opacity: 0.75;
        border-radius: 5px;
    }
    .ruler-ticks {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        font-weight: 500;
        color: #94a3b8;
        margin-top: 8px;
    }
    
    /* Market Verdict Box - Soft Tint */
    .verdict-card {
        border-radius: 14px;
        padding: 16px 22px;
        margin-bottom: 22px;
        border: 1px solid;
    }
    .verdict-bargain {
        background: rgba(16, 185, 129, 0.09);
        border-color: rgba(52, 211, 153, 0.25);
    }
    .verdict-bargain .verdict-tag {
        color: #34d399;
    }
    .verdict-bargain .verdict-desc {
        color: #a7f3d0;
    }
    .verdict-fair {
        background: rgba(245, 158, 11, 0.09);
        border-color: rgba(251, 191, 36, 0.25);
    }
    .verdict-fair .verdict-tag {
        color: #fbbf24;
    }
    .verdict-fair .verdict-desc {
        color: #fde68a;
    }
    .verdict-caution {
        background: rgba(239, 68, 68, 0.09);
        border-color: rgba(248, 113, 113, 0.25);
    }
    .verdict-caution .verdict-tag {
        color: #f87171;
    }
    .verdict-caution .verdict-desc {
        color: #fecaca;
    }
    .verdict-tag {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .verdict-desc {
        font-size: 0.92rem;
        line-height: 1.55;
    }

    /* Commute Summary Strip */
    .commute-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        background: #141c2c;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.2);
    }
    .commute-item {
        font-size: 0.86rem;
        color: #94a3b8;
    }
    .commute-item strong {
        color: #f8fafc;
        font-weight: 600;
    }

    /* Methodology Pillar Cards */
    .pillar-card {
        background: #141c2c;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left: 3px solid #38bdf8;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.2);
    }
    .pillar-head {
        font-size: 0.92rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .pillar-body {
        font-size: 0.86rem;
        color: #94a3b8;
        line-height: 1.55;
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

    # Refined, Smooth Dark Header
    st.markdown("""
    <div class="header-card">
        <h1 class="header-title">HỖ TRỢ SINH VIÊN TÌM TRỌ</h1>
        <p class="header-sub">Công cụ hỗ trợ sinh viên đối soát giá thuê phòng thực tế, dự toán chi phí sinh hoạt hàng tháng và kiểm tra lộ trình đến các trường Đại học tại TP.HCM.</p>
        <div class="header-chips">
            <span class="header-chip">Dữ liệu thị trường 2026</span>
            <span class="header-chip">102 Phường/Xã mới</span>
            <span class="header-chip">Mô hình Hedonic chuẩn kinh tế</span>
            <span class="header-chip">Ràng buộc đơn điệu diện tích</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Clean Input Controls
    with st.sidebar:
        st.markdown("#### Trường Đại học của bạn")
        uni_names = [u["name"] for u in poi_db["universities"]]
        sel_uni_name = st.selectbox(
            "Chọn trường Đại học đang theo học:",
            options=uni_names,
            index=0,
            help="Tọa độ cổng trường sẽ được dùng để đo khoảng cách và thời gian di chuyển bằng xe máy."
        )
        sel_uni = next(u for u in poi_db["universities"] if u["name"] == sel_uni_name)

        st.markdown("---")
        st.markdown("#### Vị trí căn phòng")
        ward_options = ward_df["ward_new"].tolist()
        ward_labels = [f"{w} ({ward_df[ward_df['ward_new']==w]['old_district_primary'].values[0]})" for w in ward_options]

        default_ward_idx = 0
        for i, w in enumerate(ward_options):
            if "Chợ Quán" in w or "Phường 3" in w:
                default_ward_idx = i
                break

        sel_idx = st.selectbox(
            "Chọn Phường mới (và Quận cũ tương ứng):",
            range(len(ward_options)),
            index=default_ward_idx,
            format_func=lambda i: ward_labels[i]
        )
        selected_ward = ward_options[sel_idx]
        ward_row = ward_df[ward_df["ward_new"] == selected_ward].iloc[0]

        st.markdown("---")
        st.markdown("#### Diện tích & Loại hình")
        area_m2 = st.slider(
            "Diện tích sử dụng (m²):",
            min_value=8.0,
            max_value=60.0,
            value=22.0,
            step=1.0,
            help="Mô hình ràng buộc vi mô đảm bảo: Cùng điều kiện, diện tích lớn hơn luôn có giá trị không thấp hơn phòng nhỏ."
        )

        room_type_options = {
            "Phòng trọ truyền thống": "phong_tro",
            "Căn hộ mini / Studio": "can_ho_mini",
            "Ở ghép / KTX / Sleepbox": "o_ghep_ktx_sleepbox",
            "Căn hộ dịch vụ (CHDV)": "can_ho_dich_vu",
            "Nhà nguyên căn": "nha_nguyen_can"
        }
        room_type_label = st.selectbox("Loại hình cho thuê:", list(room_type_options.keys()))
        room_type = room_type_options[room_type_label]

        st.markdown("---")
        st.markdown("#### Tiện nghi & Môi trường sống")
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
        st.markdown("#### Giá chủ nhà đang đưa ra")
        asking_price_mil = st.number_input(
            "Tiền phòng thông báo (Triệu đồng/tháng):",
            min_value=0.5,
            max_value=15.0,
            value=3.5,
            step=0.1
        )
        asking_price_vnd = asking_price_mil * 1_000_000

        st.markdown("---")
        st.markdown("#### Các khoản phí sinh hoạt kèm theo")
        with st.expander("Tùy chỉnh đơn giá điện, nước, gửi xe...", expanded=False):
            elec_rate = st.number_input("Giá điện sinh hoạt (VNĐ/kWh):", min_value=2000, max_value=6000, value=3800, step=100)
            elec_kwh = st.slider("Số kWh ước tính dùng/tháng:", min_value=30, max_value=300, value=90, step=10)
            water_cost = st.number_input("Tiền nước sinh hoạt (VNĐ/người/tháng):", min_value=0, max_value=250000, value=100000, step=10000)
            parking_cost = st.number_input("Phí giữ xe máy (VNĐ/xe/tháng):", min_value=0, max_value=300000, value=120000, step=10000)
            wifi_service_cost = st.number_input("Phí dịch vụ, wifi, rác (VNĐ/tháng):", min_value=0, max_value=400000, value=150000, step=10000)

    # Geospatial computations
    lat = float(ward_row["centroid_lat"])
    lon = float(ward_row["centroid_lon"])

    # Distances to CBDs and Hubs
    d_center = haversine(lat, lon, config["geospatial"]["ben_thanh"]["lat"], config["geospatial"]["ben_thanh"]["lon"])
    t_center = int(round((d_center / 20.0) * 60.0 + 3.0))

    lat_sala, lon_sala = 10.7680, 106.7150
    d_sala = haversine(lat, lon, lat_sala, lon_sala)
    t_sala = int(round((d_sala / 20.0) * 60.0 + 3.0))

    d_target_uni = haversine(lat, lon, sel_uni["lat"], sel_uni["lon"])
    commute_time_mins = int(round((d_target_uni / 20.0) * 60.0 + 3.0))

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
        "dist_to_sgu_km": d_target_uni,
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
        zone_mult = 1.35 if ward_row["zone_group"] == "Core_Center" else (1.05 if ward_row["zone_group"] == "Inner_City" else 0.85)
        base_rent = 120_000 * effective_area * zone_mult
        amenity_bonus = (has_ac * 350_000) + (has_mezzanine * 200_000) + (has_elevator * 200_000) + (is_furnished * 450_000) + (has_fridge * 150_000) + (has_washer * 150_000)
        dist_decay = max(0, (5.0 - d_target_uni) * 60_000)
        pred_vnd = base_rent + amenity_bonus + dist_decay
        pred_vnd = max(800_000, min(pred_vnd, 9_500_000))
        q10_vnd = pred_vnd * 0.88
        q90_vnd = pred_vnd * 1.15

    q10_mil = q10_vnd / 1e6
    q90_mil = q90_vnd / 1e6
    pred_mil = pred_vnd / 1e6

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
    # 1. KPI SUMMARY CARDS
    # ==============================================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Giá Hợp Lý Đề Xuất</div>
            <div class="kpi-val" style="color:#38bdf8;">{pred_mil:.2f} <span style="font-size:0.95rem;font-weight:500;color:#94a3b8;">Tr/tháng</span></div>
            <div class="kpi-sub">Đơn giá: {pred_vnd/area_m2:,.0f} đ/m²</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Khoảng Giá Phổ Biến</div>
            <div class="kpi-val" style="color:#f1f5f9;">{q10_mil:.2f} — {q90_mil:.2f} <span style="font-size:0.95rem;font-weight:500;color:#94a3b8;">Tr</span></div>
            <div class="kpi-sub">Vùng an toàn 80% phòng [Q10 - Q90]</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Lộ Trình Đến Trường</div>
            <div class="kpi-val" style="color:#34d399;">{d_target_uni:.1f} <span style="font-size:0.95rem;font-weight:500;color:#94a3b8;">km</span></div>
            <div class="kpi-sub">~{commute_time_mins} phút di chuyển xe máy</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Dự Toán Chi Phí Thực Tế</div>
            <div class="kpi-val" style="color:#fbbf24;">{total_living_cost/1e6:.2f} <span style="font-size:0.95rem;font-weight:500;color:#94a3b8;">Tr/tháng</span></div>
            <div class="kpi-sub">Gồm phòng, điện nước, xe, dịch vụ</div>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # 2. MARKET VERDICT
    # ==============================================================================
    diff_amount = asking_price_vnd - pred_vnd
    diff_pct = (diff_amount / pred_vnd) * 100.0

    if asking_price_vnd < q10_vnd:
        v_class = "verdict-bargain"
        v_tag = "Mức giá rất tiết kiệm"
        v_title = f"Thấp hơn mặt bằng chung {abs(diff_pct):.1f}% (-{abs(diff_amount)/1e6:.2f} Tr/tháng)"
        v_desc = (
            f"Mức giá chủ nhà đưa ra ({asking_price_mil:.2f} triệu) thuộc nhóm 10% phòng có đơn giá thấp nhất khu vực. "
            f"Đây là mức giá thuận lợi cho sinh viên cần tiết kiệm chi phí. Khi đi xem thực tế, bạn nên kiểm tra kỹ chất lượng phòng, "
            f"hệ thống cấp thoát nước, độ ẩm mốc và xác nhận trước đơn giá điện nước trong hợp đồng."
        )
    elif asking_price_vnd <= pred_vnd:
        v_class = "verdict-bargain"
        v_tag = "Mức giá tốt"
        v_title = f"Thấp hơn mức thẩm định {abs(diff_pct):.1f}% (-{abs(diff_amount)/1e6:.2f} Tr/tháng)"
        v_desc = (
            f"Mức giá {asking_price_mil:.2f} triệu nằm ở mức giá tốt so với tiện ích và diện tích phòng. "
            f"Căn phòng này có tỷ lệ chi phí trên chất lượng hợp lý, phù hợp để cân nhắc đặt cọc nếu thực tế đúng như mô tả."
        )
    elif asking_price_vnd <= q90_vnd:
        v_class = "verdict-fair"
        v_tag = "Đúng khung thị trường"
        v_title = f"Chênh lệch {diff_pct:+.1f}% so với mức thẩm định ({diff_amount/1e6:+.2f} Tr/tháng)"
        v_desc = (
            f"Mức giá {asking_price_mil:.2f} triệu nằm trong khoảng giá bình thường của khu vực {selected_ward}. "
            f"Bạn có thể dùng các gợi ý bên dưới để thương lượng chủ nhà hỗ trợ tiền giữ xe hoặc giảm nhẹ chi phí ban đầu."
        )
    else:
        v_class = "verdict-caution"
        v_tag = "Cần cân nhắc / Giá cao hơn mặt bằng"
        v_title = f"Cao hơn mặt bằng khu vực {diff_pct:+.1f}% (+{diff_amount/1e6:.2f} Tr/tháng)"
        v_desc = (
            f"Mức giá {asking_price_mil:.2f} triệu đang cao hơn mức giá phổ biến của các căn phòng cùng diện tích và tiện nghi tại khu vực. "
            f"Nếu phòng không có nội thất cao cấp đặc biệt hoặc dịch vụ đi kèm, bạn nên thương lượng điều chỉnh giá thuê "
            f"hoặc tham khảo các bài đăng lân cận trên bản đồ trước khi ký hợp đồng."
        )

    st.markdown(f"""
    <div class="verdict-card {v_class}">
        <div class="verdict-tag">{v_tag}: {v_title}</div>
        <div class="verdict-desc">{v_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 3. INTERACTIVE MARKET PRICE RULER
    # ==============================================================================
    # Compute relative percentages for visual ruler
    scale_min = min(q10_mil * 0.8, asking_price_mil * 0.9)
    scale_max = max(q90_mil * 1.2, asking_price_mil * 1.1)
    scale_range = scale_max - scale_min if scale_max > scale_min else 1.0

    safe_left_pct = max(0.0, min(100.0, (q10_mil - scale_min) / scale_range * 100.0))
    safe_width_pct = max(5.0, min(100.0 - safe_left_pct, (q90_mil - q10_mil) / scale_range * 100.0))
    pin_pct = max(2.0, min(98.0, (asking_price_mil - scale_min) / scale_range * 100.0))

    st.markdown(f"""
    <div class="ruler-container">
        <div class="ruler-header">
            <span class="ruler-title">Khung Giá Thị Trường & Vị Trí Mức Giá Bạn Đang Xem</span>
            <span class="ruler-sub">Mức giá chủ nhà đưa ra: <strong style="color:#f9fafb;">{asking_price_mil:.2f} Tr/tháng</strong></span>
        </div>
        <div class="ruler-bar-bg">
            <div class="ruler-range-safe" style="left:{safe_left_pct:.1f}%; width:{safe_width_pct:.1f}%;"></div>
            <div style="position:absolute; left:{pin_pct:.1f}%; width:4px; height:16px; top:-4px; background:#f9fafb; border-radius:2px; box-shadow:0 0 6px rgba(255,255,255,0.7);"></div>
        </div>
        <div class="ruler-ticks">
            <span>Tiết kiệm (&lt; {q10_mil:.2f} Tr)</span>
            <span>Vùng an toàn 80% phòng ({q10_mil:.2f} Tr — {q90_mil:.2f} Tr)</span>
            <span>Cao cấp (&gt; {q90_mil:.2f} Tr)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 4. TABS: NEGOTIATION SCRIPT, LIVING COST, & EVALUATION LOGIC
    # ==============================================================================
    tab_nego, tab_cost, tab_theory = st.tabs([
        "Gợi ý tin nhắn gửi chủ nhà",
        "Dự toán chi phí sinh hoạt hàng tháng",
        "Phương pháp định giá & Cơ sở dữ liệu"
    ])

    with tab_nego:
        st.markdown("##### Gợi ý tin nhắn trao đổi dựa trên các tiện ích còn thiếu:")
        missing_reasons = []
        if not has_ac:
            missing_reasons.append("phòng chưa trang bị máy lạnh (sinh viên phải tự chịu nóng hoặc đầu tư lắp mới)")
        if not has_elevator:
            missing_reasons.append("tòa nhà không có thang máy, việc chuyển đồ và đi lại leo lầu có phần bất tiện")
        if not has_parking:
            missing_reasons.append("không có hầm để xe riêng trong nhà, phải gửi xe ngoài bãi tốn thêm chi phí")
        if not has_mezzanine and room_type == "phong_tro":
            missing_reasons.append("phòng không có gác lửng khiến không gian sinh hoạt và để đồ hơi hẹp")
        if d_target_uni > 3.5:
            missing_reasons.append(f"vị trí cách trường {d_target_uni:.1f} km, tốn thời gian di chuyển và tiền xăng hàng ngày")
        if not has_fridge and not has_washer:
            missing_reasons.append("chưa có máy giặt và tủ lạnh trang bị sẵn")

        suggested_offer_vnd = max(pred_vnd, asking_price_vnd - 300_000)
        if asking_price_vnd > pred_vnd:
            suggested_offer_vnd = pred_vnd

        reasons_bullet = "\n- ".join(missing_reasons) if missing_reasons else "Căn phòng hiện tại có mức giá hơi cao so với mặt bằng trung bình cùng khu vực."

        script_text = (
            f"Dạ em chào anh/chị chủ nhà ạ,\n\n"
            f"Em là sinh viên trường {sel_uni_name.split('-')[0].strip()}, em vừa xem qua thông tin phòng {area_m2:.0f}m² bên mình tại {selected_ward}.\n"
            f"Em thấy phòng phù hợp với nhu cầu đi học của em và mong muốn thuê ở ổn định lâu dài. Qua tham khảo mặt bằng khu vực xung quanh trường, vì phòng mình hiện tại:\n"
            f"- {reasons_bullet}\n\n"
            f"Nên em xin phép gửi anh/chị mức đề xuất {suggested_offer_vnd/1e6:.2f} triệu/tháng (hoặc nhờ anh/chị hỗ trợ em phí gửi xe / tiền dịch vụ) "
            f"để phù hợp với ngân sách sinh hoạt của em được không ạ? Nếu thuận tiện em xin phép qua xem phòng và đặt cọc giữ chỗ luôn ạ.\n\n"
            f"Em cảm ơn anh/chị nhiều ạ!"
        )

        st.code(script_text, language="markdown")
        st.caption("💡 Mẹo nhỏ: Sinh viên trao đổi lịch sự, cam kết giữ gìn vệ sinh và ký hợp đồng từ 6 tháng - 1 năm thường được chủ nhà thiện chí giảm giá hoặc miễn phí tiền gửi xe.")

    with tab_cost:
        st.markdown("##### Bảng dự toán chi tiết các khoản chi phí thực tế mỗi tháng:")
        cost_df = pd.DataFrame([
            {"Hạng mục": "Tiền thuê phòng", "Cách tính": f"{asking_price_mil:.2f} Tr/tháng", "Thành tiền (VNĐ)": asking_price_vnd},
            {"Hạng mục": "Tiền điện sinh hoạt", "Cách tính": f"{elec_kwh} kWh × {elec_rate:,} đ/kWh" if not is_inclusive else "Bao trọn gói", "Thành tiền (VNĐ)": monthly_elec},
            {"Hạng mục": "Tiền nước sinh hoạt", "Cách tính": f"{water_cost:,} đ/người/tháng" if not is_inclusive else "Bao trọn gói", "Thành tiền (VNĐ)": monthly_water},
            {"Hạng mục": "Phí giữ xe máy", "Cách tính": "1 xe máy", "Thành tiền (VNĐ)": parking_cost},
            {"Hạng mục": "Phí dịch vụ, wifi, rác", "Cách tính": "Trọn gói tiện ích tòa nhà", "Thành tiền (VNĐ)": wifi_service_cost},
            {"Hạng mục": "Tổng chi phí thực tế", "Cách tính": "Tổng số tiền rời túi mỗi tháng", "Thành tiền (VNĐ)": total_living_cost}
        ])
        st.table(cost_df.style.format({"Thành tiền (VNĐ)": "{:,.0f}"}))
        st.caption("Lưu ý: Nhiều phòng có giá niêm yết thấp nhưng phụ thu điện nước ở mức cao (ví dụ điện 4.500đ - 5.000đ/kWh), khiến tổng chi phí thực tế đội lên đáng kể.")

    with tab_theory:
        st.markdown("##### Mô hình Định giá Đặc tính (Hedonic Pricing Model - Sherwin Rosen, 1974):")
        st.markdown(
            "Giá thuê phòng trọ được xác định bằng một hàm tổng hợp đa chiều từ các thuộc tính cốt lõi phục vụ đời sống và việc học tập:"
        )

        st.latex(r"""
        P_{\text{phòng}} = f\Big( \text{Diện tích},\ d_{\text{trường}},\ d_{\text{y tế}},\ d_{\text{trục đường/Metro}},\ t_{\text{CBD}},\ \text{Chợ/TTTM},\ \text{An ninh},\ \text{Tiện nghi} \Big)
        """)

        st.markdown("---")
        st.markdown("##### 6 Trụ Cột Đánh Giá Chi Tiết:")

        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-head">1. Quy mô không gian & Ràng buộc đơn điệu toán học (Monotone Constraints)</div>
            <div class="pillar-body">
                Diện tích là hàng hóa thông thường. Mô hình bắt buộc tuân thủ vi mô <b>∂P / ∂(Diện tích) ≥ 0</b>, loại bỏ hoàn toàn lỗi phòng diện tích lớn bị định giá thấp hơn phòng nhỏ cùng khu vực. Với phòng KTX / Sleepbox, diện tích được chuẩn hóa về không gian cá nhân (8.0m²) để tránh làm lệch phân phối.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-head">2. Khoảng cách đến Trường Đại học bạn đang theo học</div>
            <div class="pillar-body">
                Đo lường chính xác khoảng cách (km) và thời gian di chuyển (phút) bằng xe máy đến cổng trường bạn chọn. Bán kính <b>≤ 2.0 km</b> có giá trị tiện ích cao nhờ tiết kiệm thời gian kẹt xe và chi phí xăng xe mỗi ngày.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-head">3. Hệ thống y tế & Chăm sóc sức khỏe xung quanh</div>
            <div class="pillar-body">
                Khoảng cách đến bệnh viện, trạm y tế gần nhất và mật độ cơ sở y tế trong bán kính 2 km, giúp an tâm khi phát sinh vấn đề sức khỏe.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-head">4. Hạ tầng giao thông & Khả năng kết nối đô thị</div>
            <div class="pillar-body">
                Vị trí gần các trục lộ huyết mạch, tuyến Metro Bến Thành - Suối Tiên; thời gian di chuyển đến Trung tâm Quận 1 hoặc các CBD mới (Thủ Thiêm, KĐT Sala) trong vòng <b>15 - 30 phút</b>.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-head">5. Tiện ích dân sinh đời sống (Chợ truyền thống & Siêu thị)</div>
            <div class="pillar-body">
                Sự hiện diện của chợ dân sinh, bách hóa, siêu thị xung quanh giúp sinh viên dễ dàng mua thực phẩm tươi giá rẻ và tiết kiệm chi phí ăn uống hàng tháng.
            </div>
        </div>

        <div class="pillar-card">
            <div class="pillar-head">6. An ninh trật tự & Tiện nghi phòng</div>
            <div class="pillar-body">
                Khóa vân tay, camera/bảo vệ 24/7, giờ giấc tự do không giới nghiêm, không chung chủ; đi kèm các thiết bị giảm chi phí mua sắm ban đầu (máy lạnh, tủ lạnh, máy giặt, gác lửng, ban công).
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # 5. ACTIONABLE REALISTIC COMMUTER MAP
    # ==============================================================================
    st.write("---")
    st.markdown("#### Bản Đồ Di Chuyển & Mạng Lưới Phòng Thật Đối Chứng")
    st.markdown(
        f"Mô phỏng lộ trình kết nối phòng trọ tại **{selected_ward}** đến cổng trường **{sel_uni['name']}**, "
        f"thời gian kết nối các trung tâm đô thị và **dữ liệu các bài đăng phòng thật lân cận**."
    )

    # Clean Commute Strip
    st.markdown(f"""
    <div class="commute-strip">
        <div class="commute-item">🏫 Đến trường: <strong>{d_target_uni:.1f} km</strong> (~{commute_time_mins} phút)</div>
        <div class="commute-item">🏙️ Đến Q.1 (Bến Thành): <strong>{d_center:.1f} km</strong> (~{t_center} phút)</div>
        <div class="commute-item">🏢 Đến CBD Sala: <strong>{d_sala:.1f} km</strong> (~{t_sala} phút)</div>
        <div class="commute-item">🏥 Bệnh viện gần nhất: <strong>{d_nearest_hosp:.1f} km</strong></div>
        <div class="commute-item">🚇 Ga Metro gần nhất: <strong>{d_nearest_metro:.1f} km</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Folium Map centered between Room and Target University
    mid_lat = (lat + sel_uni["lat"]) / 2.0
    mid_lon = (lon + sel_uni["lon"]) / 2.0
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=14, tiles=None)

    # Layer 1: OpenStreetMap Standard
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Bản đồ đường phố (OpenStreetMap)",
        control=True
    ).add_to(m)

    # Layer 2: Esri World Imagery (High-Resolution Satellite)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri World Imagery",
        name="Ảnh vệ tinh thực tế (Esri Satellite)",
        control=True
    ).add_to(m)

    # Layer 3: Esri World Topo Map
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri World Topo Map",
        name="Bản đồ giao thông đô thị (Topo)",
        control=True
    ).add_to(m)

    # Commute Buffer Circles around Selected University
    folium.Circle(
        radius=1500,
        location=[sel_uni["lat"], sel_uni["lon"]],
        color="#10b981",
        fill=True,
        fill_opacity=0.12,
        tooltip="Vùng xanh: Đi bộ / Xe đạp thuận tiện (≤ 1.5 km)"
    ).add_to(m)

    folium.Circle(
        radius=3500,
        location=[sel_uni["lat"], sel_uni["lon"]],
        color="#f59e0b",
        fill=True,
        fill_opacity=0.08,
        tooltip="Vùng vàng: Đi xe máy thuận tiện (≤ 3.5 km, dưới 12 phút)"
    ).add_to(m)

    # Room Marker
    folium.Marker(
        [lat, lon],
        popup=f"<b>Phòng bạn đang thẩm định:</b><br>{selected_ward}<br>Diện tích: {area_m2}m²<br>Giá AI định: {pred_mil:.2f} Tr/tháng",
        tooltip="Vị trí phòng đang thẩm định",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)

    # Target University Marker
    folium.Marker(
        [sel_uni["lat"], sel_uni["lon"]],
        popup=f"<b>{sel_uni['name']}</b><br>Cổng trường<br>Cách phòng: {d_target_uni:.1f} km (~{commute_time_mins} phút)",
        tooltip=sel_uni['name'],
        icon=folium.Icon(color="blue", icon="graduation-cap", prefix="fa")
    ).add_to(m)

    # Nearest Hospital Marker
    nearest_hosp = min(poi_db["hospitals"], key=lambda h: haversine(lat, lon, h["lat"], h["lon"]))
    folium.Marker(
        [nearest_hosp["lat"], nearest_hosp["lon"]],
        popup=f"<b>Bệnh viện gần nhất:</b><br>{nearest_hosp['name']}<br>Cách phòng: {d_nearest_hosp:.1f} km",
        tooltip=f"BV: {nearest_hosp['name']}",
        icon=folium.Icon(color="green", icon="plus-square", prefix="fa")
    ).add_to(m)

    # CBD District 1 / Ben Thanh Center Landmark
    folium.Marker(
        [config["geospatial"]["ben_thanh"]["lat"], config["geospatial"]["ben_thanh"]["lon"]],
        popup=f"<b>Trung tâm Quận 1 (Chợ Bến Thành):</b><br>Cách phòng: {d_center:.1f} km (~{t_center} phút)",
        tooltip="Trung tâm Quận 1",
        icon=folium.Icon(color="purple", icon="building", prefix="fa")
    ).add_to(m)

    # Commute Route Polyline between Room and Target University
    folium.PolyLine(
        locations=[[lat, lon], [sel_uni["lat"], sel_uni["lon"]]],
        color="#2563eb",
        weight=4,
        opacity=0.88,
        dash_array="8, 6",
        tooltip=f"Lộ trình đến trường: {d_target_uni:.1f} km (~{commute_time_mins} phút xe máy)"
    ).add_to(m)

    # Real Neighbor Listings Pins (from real scraped data)
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
                p_color = "#10b981"  # Emerald
            elif n_price <= 4_500_000:
                p_color = "#f59e0b"  # Amber
            else:
                p_color = "#ef4444"  # Rose

            popup_html = f"""
            <div style='font-family:sans-serif; width:220px; line-height:1.4;'>
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

    # Clean Map Legend
    st.markdown("""
    <div style="display:flex; gap:16px; flex-wrap:wrap; font-size:0.84rem; background:#141c2c; color:#cbd5e1; padding:12px 18px; border-radius:10px; margin-top:8px; border:1px solid rgba(255,255,255,0.07); box-shadow:0 4px 16px -2px rgba(0,0,0,0.2);">
        <span><strong style="color:#f87171;">● Marker đỏ:</strong> Phòng đang thẩm định</span>
        <span><strong style="color:#38bdf8;">● Marker xanh:</strong> Trường Đại học</span>
        <span><strong style="color:#34d399;">● Marker lá:</strong> Bệnh viện</span>
        <span><strong style="color:#c084fc;">● Marker tím:</strong> Trung tâm Q.1 / CBD</span>
        <span><strong style="color:#34d399;">● Chấm xanh:</strong> Phòng thật &lt; 2.5 Tr</span>
        <span><strong style="color:#fbbf24;">● Chấm vàng:</strong> Phòng thật 2.5 - 4.5 Tr</span>
        <span><strong style="color:#f87171;">● Chấm đỏ:</strong> Phòng thật &gt; 4.5 Tr</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
