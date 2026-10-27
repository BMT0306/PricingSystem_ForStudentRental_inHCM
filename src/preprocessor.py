"""
Module: src/preprocessor.py
Description: Data cleaning, unit parsing, domain constraints, dual-layer geocoding,
geospatial POI distance calculations, regex NLP feature extraction, grouped IQR outlier
removal, and zero-leakage stratified train/val/test splitting.
"""

import os
import re
import math
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_poi_database(poi_path: str = "config/poi_database.yaml") -> dict:
    with open(poi_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# ==============================================================================
# 1. UNIT PARSING FUNCTIONS
# ==============================================================================

def parse_price_vnd(price_raw: Optional[str], price_json: Optional[any] = None) -> Optional[float]:
    """
    Convert price to Vietnamese Dong (VNĐ/month).
    Handles:
    - '3.5 triệu/tháng', '3,5 tr', '3tr5' -> 3,500,000
    - '850 nghìn/tháng', '850k' -> 850,000
    - '4.500.000 đ/tháng' -> 4,500,000
    - 'Thỏa thuận', 'Liên hệ', per day -> None
    """
    # If price_json is an integer/float already in VND
    if price_json is not None:
        try:
            val = float(price_json)
            if 100_000 <= val <= 50_000_000:
                return val
        except (ValueError, TypeError):
            pass

    if not price_raw or not isinstance(price_raw, str):
        return None

    text = price_raw.strip().lower()

    # Drop non-standard prices
    if any(k in text for k in ["thỏa thuận", "thoa thuan", "liên hệ", "lien he", "đ/ngày", "ngay", "đêm"]):
        return None

    # Pattern: 3.5 triệu or 3,5 triệu or 3tr5
    # Match "3tr5"
    m_tr_split = re.search(r"(\d+)\s*(?:tr|triệu)\s*(\d+)", text)
    if m_tr_split:
        whole = float(m_tr_split.group(1))
        dec = float(m_tr_split.group(2))
        return (whole + dec / 10.0) * 1_000_000

    # Match "3.5 triệu" or "3,5 tr"
    m_trieu = re.search(r"([\d\.,]+)\s*(?:triệu|trieu|tr)", text)
    if m_trieu:
        num_str = m_trieu.group(1).replace(",", ".")
        try:
            return float(num_str) * 1_000_000
        except ValueError:
            pass

    # Match "850 nghìn" or "850 ngàn" or "850k"
    m_nghin = re.search(r"([\d\.,]+)\s*(?:nghìn|nghin|ngàn|ngan|k)", text)
    if m_nghin:
        num_str = m_nghin.group(1).replace(",", ".")
        try:
            return float(num_str) * 1_000
        except ValueError:
            pass

    # Match exact numbers: "3.500.000 đ" or "3500000"
    m_exact = re.search(r"([\d\.,]+)\s*(?:đ|vnđ|vnd)?", text)
    if m_exact:
        num_clean = re.sub(r"[\.,]", "", m_exact.group(1))
        try:
            val = float(num_clean)
            if 100_000 <= val <= 50_000_000:
                return val
        except ValueError:
            pass

    return None


def parse_area_m2(area_raw: Optional[str]) -> Optional[float]:
    """
    Extract area in square meters (float).
    Example: '25.5 m2', '30m²', '20 m' -> float
    """
    if not area_raw or not isinstance(area_raw, str):
        return None

    text = area_raw.strip().lower()
    m = re.search(r"([\d\.,]+)\s*(?:m2|m²|m)", text)
    if m:
        num_str = m.group(1).replace(",", ".")
        try:
            return float(num_str)
        except ValueError:
            pass
    return None


# ==============================================================================
# 2. WARD NORMALIZATION & GEOCODING ENGINE
# ==============================================================================

class GeospatialEngine:
    def __init__(self, ward_mapping_csv: str = "data/reference/ward_mapping_hcm.csv", cache_db: str = "data/reference/nominatim_cache.sqlite"):
        self.ward_df = pd.read_csv(ward_mapping_csv, encoding="utf-8")
        self.cache_db = cache_db
        self._init_sqlite()

        # Build ward lookup dictionary
        self.ward_lookup = {}
        for _, row in self.ward_df.iterrows():
            w_name = str(row["ward_new"]).strip()
            # Canonical key: remove 'phường', 'xã', 'thị trấn'
            clean_key = re.sub(r"^(phường|xã|thị trấn)\s+", "", w_name.lower())
            self.ward_lookup[clean_key] = row.to_dict()

    def _init_sqlite(self):
        os.makedirs(os.path.dirname(self.cache_db), exist_ok=True)
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS geocache (
                    address_query TEXT PRIMARY KEY,
                    lat REAL,
                    lon REAL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def match_ward(self, address_text: str) -> Optional[dict]:
        """
        Match address text to one of the 102 canonical new wards.
        """
        if not address_text or not isinstance(address_text, str):
            return None

        addr_lower = address_text.lower()
        for key, row_data in self.ward_lookup.items():
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, addr_lower):
                return row_data
        return None

    def get_coordinates(self, address: str, ward_row: dict) -> Tuple[float, float, str]:
        """
        Dual-layer coordinate retrieval:
        1. Check SQLite cache
        2. Fallback to ward centroid
        """
        # Try checking cache
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT lat, lon FROM geocache WHERE address_query = ?", (address.strip(),))
            res = cursor.fetchone()
            if res and res[0] is not None:
                return float(res[0]), float(res[1]), "street"

        # Fallback to ward centroid
        lat = float(ward_row.get("centroid_lat", 10.7725))
        lon = float(ward_row.get("centroid_lon", 106.6980))
        return lat, lon, "ward_centroid"


# ==============================================================================
# 3. NLP REGEX FEATURE EXTRACTION
# ==============================================================================

def extract_nlp_features(row: pd.Series) -> Dict[str, any]:
    """
    Extract 15+ binary and categorical room features by combining:
    - Scraped checkbox indicators (Layer 2)
    - Vietnamese regex keyword matching on (title + description)
    """
    title = str(row.get("title", "") or "")
    desc = str(row.get("full_description", "") or "")
    combined_text = (title + " " + desc).lower()

    def check(checkbox_col: Optional[str], regex_pattern: str) -> int:
        if checkbox_col and checkbox_col in row and row[checkbox_col] == 1:
            return 1
        if re.search(regex_pattern, combined_text):
            return 1
        return 0

    features = {}

    # Core amenities
    features["has_ac"] = check("co_may_lanh", r"máy lạnh|may lanh|điều hòa|dieu hoa|\bac\b")
    features["has_mezzanine"] = check("co_gac", r"gác|gac|lửng|lung|duplex")
    features["has_private_wc"] = check(None, r"wc riêng|wc rieng|toilet riêng|toilet rieng|vệ sinh riêng|ve sinh rieng|khép kín|khep kin")
    features["has_kitchen"] = check("co_ke_bep", r"kệ bếp|ke bep|bếp riêng|bep rieng|nấu ăn|nau an")
    features["has_balcony_window"] = check(None, r"ban công|ban cong|cửa sổ|cua so|thoáng khí|giếng trời")
    features["has_fridge"] = check("co_tu_lanh", r"tủ lạnh|tu lanh")
    features["has_washer"] = check("co_may_giat", r"máy giặt|may giat")
    features["has_elevator"] = check("co_thang_may", r"thang máy|thang may")
    features["is_fully_furnished"] = check("day_du_noi_that", r"đầy đủ nội thất|full nội thất|full noi that|nội thất cao cấp")

    # Freedom & Security
    features["no_curfew"] = check("gio_giac_tu_do", r"giờ giấc tự do|gio giac tu do|không giới nghiêm|24/24")
    features["not_shared_owner"] = check("khong_chung_chu", r"không chung chủ|khong chung chu|riêng chủ")
    features["has_security"] = check("co_bao_ve_24_24", r"bảo vệ|bao ve|khóa vân tay|khoa van tay|camera")
    features["has_parking_basement"] = check("co_ham_de_xe", r"hầm để xe|ham de xe|chỗ để xe|bãi xe")

    # Hidden cost & Student specific
    features["is_all_inclusive"] = check(None, r"bao điện nước|bao dien nuoc|trọn gói|tron goi|bao trọn gói|miễn phí điện nước")
    features["female_only"] = check(None, r"chỉ nhận nữ|chi nhan nu|ở ghép nữ|o ghep nu|ktx nữ|phòng nữ")
    features["is_shared_dorm"] = check(None, r"ở ghép|o ghep|ký túc xá|ky tuc xa|\bktx\b|sleepbox|sleep box|giường tầng")

    # Composite amenity score (sum of positive features)
    amenity_cols = [
        "has_ac", "has_mezzanine", "has_private_wc", "has_kitchen",
        "has_balcony_window", "has_fridge", "has_washer", "has_elevator",
        "is_fully_furnished", "no_curfew", "not_shared_owner", "has_security",
        "has_parking_basement"
    ]
    features["amenity_score"] = sum(features[c] for c in amenity_cols)
    features["text_word_count"] = len(combined_text.split())

    # Room Type Classification
    if features["is_shared_dorm"] == 1:
        features["room_type"] = "o_ghep_ktx_sleepbox"
    elif "căn hộ mini" in combined_text or "chung cư mini" in combined_text or "can ho mini" in combined_text:
        features["room_type"] = "can_ho_mini"
    elif "căn hộ dịch vụ" in combined_text or "chdv" in combined_text:
        features["room_type"] = "can_ho_dich_vu"
    elif "nguyên căn" in combined_text or "nguyen can" in combined_text:
        features["room_type"] = "nha_nguyen_can"
    else:
        features["room_type"] = "phong_tro"

    return features


# ==============================================================================
# 4. MASTER PREPROCESSING PIPELINE
# ==============================================================================

class DataPreprocessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.poi_db = load_poi_database(self.config["geospatial"]["poi_database_yaml"])
        self.geo_engine = GeospatialEngine(
            ward_mapping_csv=self.config["geospatial"]["ward_mapping_csv"],
            cache_db=self.config["geospatial"]["cache_db"]
        )

    def process_raw_dataset(self, raw_csv_path: Optional[str] = None) -> pd.DataFrame:
        raw_path = raw_csv_path or self.config["data_paths"]["raw"]
        logger.info(f"Loading raw dataset from {raw_path}")
        df = pd.read_csv(raw_path, encoding="utf-8")
        n_initial = len(df)
        logger.info(f"Initial raw rows: {n_initial}")

        # 1. Deduplication by URL and (title, price_raw, area_raw)
        df = df.drop_duplicates(subset=["url"]).copy()
        if "phone_hash" in df.columns:
            df = df.drop_duplicates(subset=["phone_hash", "price_raw", "area_raw"]).copy()
        logger.info(f"After deduplication: {len(df)} rows")

        # 2. Parse Unit Features: price_vnd & area_m2
        df["price_vnd"] = [parse_price_vnd(p_raw, p_json) for p_raw, p_json in zip(df.get("price_raw", []), df.get("price_json", []))]
        df["area_m2"] = df["area_raw"].apply(parse_area_m2)

        # 3. Hard Domain Filter: Area & Price
        min_area = self.config["domain_constraints"]["min_area_m2"]
        max_area = self.config["domain_constraints"]["max_area_m2"]
        min_price = self.config["domain_constraints"]["min_price_vnd"]
        max_price = self.config["domain_constraints"]["max_price_vnd"]

        df = df.dropna(subset=["price_vnd", "area_m2"]).copy()
        df = df[(df["area_m2"] >= min_area) & (df["area_m2"] <= max_area)]
        df = df[(df["price_vnd"] >= min_price) & (df["price_vnd"] <= max_price)]
        logger.info(f"After area [8, 60]m2 & price [500k, 10M] filtering: {len(df)} rows")

        # 4. Hard Temporal Filter: Year >= 2026
        min_year = self.config["crawler"]["year_constraint"]
        if "post_year" in df.columns:
            df = df[df["post_year"] >= min_year].copy()
        logger.info(f"After year >= {min_year} filter: {len(df)} rows")

        # 5. Ward Matching & Geospatial Features
        matched_records = []
        for idx, row in df.iterrows():
            addr = str(row.get("address_raw", "")) + " " + str(row.get("title", ""))
            ward_info = self.geo_engine.match_ward(addr)
            if not ward_info:
                # If cannot match from address, try old_district fallback from URL
                continue

            lat, lon, precision = self.geo_engine.get_coordinates(row.get("address_raw", ""), ward_info)

            # Compute POI distances
            # Distance to Ben Thanh Center
            d_center = haversine_distance(
                lat, lon,
                self.config["geospatial"]["ben_thanh"]["lat"],
                self.config["geospatial"]["ben_thanh"]["lon"]
            )
            # Distance to SGU Main Campus
            d_sgu = haversine_distance(
                lat, lon,
                self.config["geospatial"]["sgu_main_campus"]["lat"],
                self.config["geospatial"]["sgu_main_campus"]["lon"]
            )

            # Nearest university & count within 2km
            uni_dists = [haversine_distance(lat, lon, u["lat"], u["lon"]) for u in self.poi_db["universities"]]
            d_nearest_uni = min(uni_dists)
            n_uni_2km = sum(1 for d in uni_dists if d <= 2.0)

            # Nearest hospital & count within 2km
            hosp_dists = [haversine_distance(lat, lon, h["lat"], h["lon"]) for h in self.poi_db["hospitals"]]
            d_nearest_hosp = min(hosp_dists)
            n_hosp_2km = sum(1 for d in hosp_dists if d <= 2.0)

            # Nearest metro station
            metro_dists = [haversine_distance(lat, lon, m["lat"], m["lon"]) for m in self.poi_db["metro_stations"]]
            d_nearest_metro = min(metro_dists)

            nlp_feats = extract_nlp_features(row)

            # Domain normalization: In shared dormitories (KTX / Sleepbox), landlords post total floor area (e.g. 45-60m2)
            # while the rental price is for a single bed. Adjust to personal usable space (8.0 m2) to prevent distortion!
            area_val = float(row["area_m2"])
            if nlp_feats.get("is_shared_dorm") == 1 and area_val > 16.0:
                area_val = 8.0

            record = {
                "post_id": row.get("post_id"),
                "url": row.get("url"),
                "title": row.get("title"),
                "price_vnd": row["price_vnd"],
                "area_m2": area_val,
                "price_per_m2": row["price_vnd"] / area_val,
                "ward_new": ward_info["ward_new"],
                "old_district": ward_info["old_district_primary"],
                "zone_group": ward_info["zone_group"],
                "lat": lat,
                "lon": lon,
                "geo_precision": precision,
                "dist_to_center_km": round(d_center, 2),
                "dist_to_sgu_km": round(d_sgu, 2),
                "dist_nearest_uni_km": round(d_nearest_uni, 2),
                "n_uni_within_2km": n_uni_2km,
                "dist_nearest_hospital_km": round(d_nearest_hosp, 2),
                "n_hospitals_within_2km": n_hosp_2km,
                "dist_nearest_metro_km": round(d_nearest_metro, 2),
                "post_date": row.get("post_date"),
                **nlp_feats
            }
            matched_records.append(record)

        clean_df = pd.DataFrame(matched_records)
        logger.info(f"After Ward & POI Geospatial Matching: {len(clean_df)} rows")

        # 6. Grouped IQR Outlier Removal by old_district
        iqr_cleaned_records = []
        for dist, group in clean_df.groupby("old_district"):
            if len(group) < 5:
                iqr_cleaned_records.append(group)
                continue
            q1 = group["price_per_m2"].quantile(0.25)
            q3 = group["price_per_m2"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered = group[(group["price_per_m2"] >= lower_bound) & (group["price_per_m2"] <= upper_bound)]
            iqr_cleaned_records.append(filtered)

        final_clean_df = pd.concat(iqr_cleaned_records, ignore_index=True)
        logger.info(f"After Grouped IQR Outlier Removal: {len(final_clean_df)} rows")

        # Save cleaned full dataset
        cleaned_path = self.config["data_paths"]["cleaned"]
        os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
        final_clean_df.to_csv(cleaned_path, index=False, encoding="utf-8")
        logger.info(f"Saved cleaned dataset to {cleaned_path}")

        # 7. Stratified Train / Val / Test Split (70 / 15 / 15)
        self.split_and_save(final_clean_df)

        return final_clean_df

    def split_and_save(self, df: pd.DataFrame):
        seed = self.config["project"]["seed"]
        strat_col = self.config["splits"]["stratify_col"]

        # Handle rare categories so that every class has at least 3 instances for 70/15/15 split
        counts = df[strat_col].value_counts()
        rare_classes = counts[counts < 3].index
        most_common = counts.index[0]
        strat_series = df[strat_col].apply(lambda x: most_common if x in rare_classes else x)

        # 70% Train, 30% Temp (Val + Test)
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=seed,
            stratify=strat_series
        )

        # Split Temp into 50% Val and 50% Test (each is 15% of total)
        temp_strat_series = strat_series.loc[temp_df.index]
        temp_counts = temp_strat_series.value_counts()
        temp_rare = temp_counts[temp_counts < 2].index
        temp_most_common = temp_counts.index[0]
        temp_strat_final = temp_strat_series.apply(lambda x: temp_most_common if x in temp_rare else x)

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=seed,
            stratify=temp_strat_final
        )

        # Assert zero leakage
        assert set(train_df.index).isdisjoint(set(test_df.index)), "Leakage detected between Train and Test!"
        assert set(train_df.index).isdisjoint(set(val_df.index)), "Leakage detected between Train and Val!"

        # Save splits
        train_df.to_csv(self.config["data_paths"]["train"], index=False, encoding="utf-8")
        val_df.to_csv(self.config["data_paths"]["val"], index=False, encoding="utf-8")
        test_df.to_csv(self.config["data_paths"]["test"], index=False, encoding="utf-8")

        logger.info(f"Saved stratified splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process_raw_dataset()
