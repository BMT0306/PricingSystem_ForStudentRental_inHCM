"""
Module: src/crawler.py
Description: Multi-endpoint resilient crawler for phongtro123.com with 2-layer parsing,
checkpointing, auto-resume, and hard temporal constraint (Year >= 2026).
"""

import os
import re
import time
import json
import random
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import yaml
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crawler.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_post_date(date_raw: str, reference_date: Optional[datetime] = None) -> Optional[datetime]:
    """
    Parse raw Vietnamese date string into a datetime object.
    Supports:
    - Absolute dates: '25/10/2026', '2026-10-25', 'Chủ Nhật, 13:20 25/10/2026'
    - Relative dates: 'vừa xong', 'hôm nay', 'hôm qua', 'X ngày trước', 'X giờ trước', 'X phút trước'
    Returns None if date cannot be parsed.
    """
    if not date_raw or not isinstance(date_raw, str):
        return None

    now = reference_date or datetime.now()
    text = date_raw.strip().lower()

    # 1. Try finding absolute date dd/mm/yyyy
    abs_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if abs_match:
        d, m, y = int(abs_match.group(1)), int(abs_match.group(2)), int(abs_match.group(3))
        try:
            # Also extract time if present (hh:mm)
            time_match = re.search(r"(\d{1,2}):(\d{2})", text)
            hour, minute = (int(time_match.group(1)), int(time_match.group(2))) if time_match else (0, 0)
            return datetime(y, m, d, hour, minute)
        except ValueError:
            pass

    # 2. Try ISO format yyyy-mm-dd
    iso_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if iso_match:
        y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
        try:
            return datetime(y, m, d)
        except ValueError:
            pass

    # 3. Handle relative times
    if "vừa xong" in text or "vua xong" in text:
        return now

    if "hôm nay" in text or "hom nay" in text:
        time_match = re.search(r"(\d{1,2}):(\d{2})", text)
        hour, minute = (int(time_match.group(1)), int(time_match.group(2))) if time_match else (now.hour, now.minute)
        return datetime(now.year, now.month, now.day, hour, minute)

    if "hôm qua" in text or "hom qua" in text:
        yesterday = now - timedelta(days=1)
        time_match = re.search(r"(\d{1,2}):(\d{2})", text)
        hour, minute = (int(time_match.group(1)), int(time_match.group(2))) if time_match else (12, 0)
        return datetime(yesterday.year, yesterday.month, yesterday.day, hour, minute)

    # X phút trước
    min_match = re.search(r"(\d+)\s*(phút|phut)", text)
    if min_match:
        return now - timedelta(minutes=int(min_match.group(1)))

    # X giờ trước
    hr_match = re.search(r"(\d+)\s*(giờ|gio)", text)
    if hr_match:
        return now - timedelta(hours=int(hr_match.group(1)))

    # X ngày trước
    day_match = re.search(r"(\d+)\s*(ngày|ngay)", text)
    if day_match:
        return now - timedelta(days=int(day_match.group(1)))

    # X tuần trước
    week_match = re.search(r"(\d+)\s*(tuần|tuan)", text)
    if week_match:
        return now - timedelta(weeks=int(week_match.group(1)))

    # X tháng trước
    month_match = re.search(r"(\d+)\s*(tháng|thang)", text)
    if month_match:
        return now - timedelta(days=int(month_match.group(1)) * 30)

    return None


def sanitize_text(text: Optional[str]) -> str:
    """Strip newlines, carriage returns, tabs, and excess whitespace to keep CSV strictly 1-line per row."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class PhongTro123Crawler:
    """
    Robust 2-Layer Crawler for phongtro123.com with checkpointing and year filter.
    """

    DISTRICT_ENDPOINTS = [
        "quan-1", "quan-2", "quan-3", "quan-4", "quan-5", "quan-6",
        "quan-7", "quan-8", "quan-9", "quan-10", "quan-11", "quan-12",
        "quan-binh-thanh", "quan-binh-tan", "quan-go-vap", "quan-phu-nhuan",
        "quan-tan-binh", "quan-tan-phu", "quan-thu-duc", "huyen-binh-chanh",
        "huyen-can-gio", "huyen-cu-chi", "huyen-hoc-mon", "huyen-nha-be"
    ]

    CATEGORY_ENDPOINTS = [
        "cho-thue-phong-tro",
        "cho-thue-can-ho-mini",
        "tim-nguoi-o-ghep",
        "cho-thue-can-ho-dich-vu",
        "nha-cho-thue"
    ]

    AMENITY_KEYS = [
        "day_du_noi_that", "co_gac", "co_ke_bep", "co_may_lanh",
        "co_may_giat", "co_tu_lanh", "co_thang_may", "khong_chung_chu",
        "gio_giac_tu_do", "co_bao_ve_24_24", "co_ham_de_xe"
    ]

    AMENITY_MAPPING = {
        "đầy đủ nội thất": "day_du_noi_that",
        "có gác": "co_gac",
        "có kệ bếp": "co_ke_bep",
        "có máy lạnh": "co_may_lanh",
        "có máy giặt": "co_may_giat",
        "có tủ lạnh": "co_tu_lanh",
        "có thang máy": "co_thang_may",
        "không chung chủ": "khong_chung_chu",
        "giờ giấc tự do": "gio_giac_tu_do",
        "có bảo vệ 24/24": "co_bao_ve_24_24",
        "có hầm để xe": "co_ham_de_xe",
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.base_url = self.config["crawler"]["base_url"]
        self.target_count = self.config["crawler"]["target_raw_count"]
        self.year_constraint = self.config["crawler"]["year_constraint"]
        self.early_stop_count = self.config["crawler"]["early_stop_consecutive_old"]
        self.delay_range = tuple(self.config["crawler"]["delay_range"])
        self.checkpoint_interval = self.config["crawler"]["checkpoint_interval"]
        self.raw_data_path = self.config["crawler"]["raw_data_path"]
        self.user_agents = self.config["crawler"]["user_agents"]

        self.session = requests.Session()
        self.crawled_urls: Set[str] = set()
        self.crawled_ids: Set[str] = set()
        self.scraped_records: List[dict] = []

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        self._load_checkpoint()

    def _get_random_headers(self) -> dict:
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://phongtro123.com/",
            "Connection": "keep-alive"
        }

    def _polite_delay(self):
        time.sleep(random.uniform(*self.delay_range))

    def _load_checkpoint(self):
        """Load already crawled items from CSV checkpoint to enable resume."""
        if not os.path.exists(self.raw_data_path):
            logger.info("No existing checkpoint found. Starting fresh.")
            return

        import pandas as pd
        try:
            df = pd.read_csv(self.raw_data_path, encoding="utf-8")
            if "url" in df.columns:
                self.crawled_urls = set(df["url"].dropna().tolist())
            if "post_id" in df.columns:
                self.crawled_ids = set(df["post_id"].dropna().astype(str).tolist())
            logger.info(f"Loaded checkpoint with {len(self.crawled_urls)} existing URLs.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    def save_checkpoint(self):
        """Append newly scraped records to raw CSV."""
        if not self.scraped_records:
            return

        import pandas as pd
        new_df = pd.DataFrame(self.scraped_records)
        file_exists = os.path.exists(self.raw_data_path)

        new_df.to_csv(
            self.raw_data_path,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
            encoding="utf-8",
            lineterminator="\n"
        )
        logger.info(f"Saved {len(self.scraped_records)} items to {self.raw_data_path}. Total collected: {len(self.crawled_urls)}")
        self.scraped_records.clear()

    def fetch_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Fetch HTML content with retry and backoff."""
        for attempt in range(max_retries):
            try:
                self._polite_delay()
                resp = self.session.get(url, headers=self._get_random_headers(), timeout=12)
                if resp.status_code == 200:
                    return resp.text
                elif resp.status_code in (404, 410):
                    logger.warning(f"Page not found ({resp.status_code}): {url}")
                    return None
                else:
                    logger.warning(f"HTTP {resp.status_code} for {url}, attempt {attempt+1}/{max_retries}")
            except Exception as e:
                logger.warning(f"Request error for {url}: {e}, attempt {attempt+1}/{max_retries}")
            time.sleep(1.5 * (attempt + 1))
        return None

    def parse_listing_items(self, html: str) -> List[dict]:
        """
        Layer 1 Parser: Extract listing metadata from JSON-LD and HTML elements.
        """
        soup = BeautifulSoup(html, "html.parser")
        items = []

        listing_container = soup.find("ul", class_="post__listing")
        if not listing_container:
            return items

        for li in listing_container.find_all("li", recursive=False):
            # 1. Try extracting structured JSON-LD first
            json_ld_script = li.find("script", type="application/ld+json")
            json_data = {}
            if json_ld_script and json_ld_script.string:
                try:
                    json_data = json.loads(json_ld_script.string)
                except Exception:
                    pass

            # URL
            url = json_data.get("url")
            if not url:
                a_tag = li.find("a", href=True)
                if a_tag:
                    url = urljoin(self.base_url, a_tag["href"])

            if not url:
                continue

            # Extract post_id from url or save button
            post_id = None
            save_btn = li.find("button", class_=re.compile(r"js-btn-save|btn__save"))
            if save_btn and save_btn.get("data-post-id"):
                post_id = save_btn["data-post-id"]
            elif url:
                id_match = re.search(r"pr(\d+)\.html", url)
                if id_match:
                    post_id = id_match.group(1)

            # Title
            title = json_data.get("name")
            if not title:
                h3_tag = li.find(["h3", "h4"])
                title = h3_tag.get_text(strip=True) if h3_tag else ""

            # Price raw & numeric
            price_raw = ""
            price_val = json_data.get("priceRange")
            price_span = li.find("span", class_=re.compile(r"text-green|post-price"))
            if price_span:
                price_raw = price_span.get_text(strip=True)

            # Area raw
            area_raw = ""
            for span in li.find_all("span"):
                txt = span.get_text(strip=True)
                if "m2" in txt or "m²" in txt:
                    area_raw = txt
                    break

            # Address / Location raw
            address_raw = ""
            address_data = json_data.get("address", {})
            if isinstance(address_data, dict):
                address_raw = address_data.get("streetAddress", "")
            if not address_raw:
                loc_a = li.find("a", href=re.compile(r"/quan-|\.html"))
                if loc_a:
                    address_raw = loc_a.get_text(strip=True)

            # Phone (one-way hash)
            phone_hash = ""
            phone = json_data.get("telephone", "")
            if not phone:
                phone_btn = li.find("button", class_=re.compile(r"btn-green"))
                if phone_btn:
                    phone = phone_btn.get_text(strip=True)
            if phone:
                clean_phone = re.sub(r"\D", "", phone)
                phone_hash = hashlib.sha256(clean_phone.encode()).hexdigest()[:12]

            # Date posted
            time_tag = li.find("time")
            date_raw = ""
            if time_tag:
                date_raw = time_tag.get("title") or time_tag.get("aria-label") or time_tag.get_text(strip=True)

            post_dt = parse_post_date(date_raw)

            items.append({
                "post_id": post_id,
                "url": url,
                "title": sanitize_text(title),
                "price_raw": sanitize_text(price_raw),
                "price_json": price_val,
                "area_raw": sanitize_text(area_raw),
                "address_raw": sanitize_text(address_raw),
                "phone_hash": phone_hash,
                "date_raw": sanitize_text(date_raw),
                "post_date": post_dt.isoformat() if post_dt else None,
                "post_year": post_dt.year if post_dt else None
            })

        return items

    AMENITY_REGEX = {
        "day_du_noi_that": r"đầy đủ nội thất|full nội thất|full noi that|nội thất cao cấp|đầy đủ tiện nghi",
        "co_gac": r"gác|gac|lửng|lung|duplex",
        "co_ke_bep": r"kệ bếp|ke bep|bếp riêng|bep rieng|nấu ăn|nau an|khu bếp|bếp ga|bếp từ",
        "co_may_lanh": r"máy lạnh|may lanh|điều hòa|dieu hoa|\bac\b",
        "co_may_giat": r"máy giặt|may giat",
        "co_tu_lanh": r"tủ lạnh|tu lanh",
        "co_thang_may": r"thang máy|thang may",
        "khong_chung_chu": r"không chung chủ|khong chung chu|riêng chủ|giờ giấc tự do",
        "gio_giac_tu_do": r"giờ giấc tự do|gio giac tu do|không giới nghiêm|24/24",
        "co_bao_ve_24_24": r"bảo vệ|bao ve|khóa vân tay|khoa van tay|camera",
        "co_ham_de_xe": r"hầm để xe|ham de xe|chỗ để xe|bãi xe|nhà để xe"
    }

    def is_valid_rental_listing(self, item: dict) -> bool:
        """
        Validate domain constraints:
        - Temporal: Year >= 2026
        - Price: 500,000 to 10,000,000 VND
        - Area: 8.0 to 60.0 m2
        """
        # 1. Year check
        post_year = item.get("post_year")
        if post_year is not None and post_year < self.year_constraint:
            return False

        # 2. Price check
        price_val = item.get("price_json")
        if price_val is not None:
            try:
                p = float(price_val)
                if p < 500000 or p > 10000000:
                    return False
            except Exception:
                pass

        # 3. Area check
        area_raw = item.get("area_raw", "")
        m = re.search(r"([\d\.]+)", str(area_raw))
        if m:
            try:
                a = float(m.group(1))
                if a < 8.0 or a > 60.0:
                    return False
            except Exception:
                pass

        return True

    def parse_detail_page(self, html: str, title: str = "") -> dict:
        """
        Layer 2 Parser: Extract 11 amenity checkboxes, full description text,
        and enrich amenities combining HTML checkboxes with title/description NLP regex.
        """
        soup = BeautifulSoup(html, "html.parser")
        detail = {
            "full_description": "",
            "confirmed_post_date": None,
            "category_name": ""
        }
        for k in self.AMENITY_KEYS:
            detail[k] = 0

        # 1. Full Description
        desc_h2 = soup.find(lambda tag: tag.name in ["h2", "h3"] and "thông tin mô tả" in tag.get_text().lower())
        if desc_h2:
            desc_texts = []
            for sibling in desc_h2.next_siblings:
                if getattr(sibling, "name", None) in ["h2", "h3"]:
                    break
                if hasattr(sibling, "get_text"):
                    t = sibling.get_text(separator="\n", strip=True)
                    if t:
                        desc_texts.append(t)
            if desc_texts:
                detail["full_description"] = sanitize_text(" ".join(desc_texts))

        if not detail["full_description"]:
            section = soup.find("section", class_=re.compile(r"post-description|post__content"))
            if section:
                detail["full_description"] = sanitize_text(section.get_text(separator=" ", strip=True))

        # 2. Extract Amenity Checkboxes under "Nổi bật"
        highlight_h2 = soup.find(lambda tag: tag.name in ["h2", "h3"] and "nổi bật" in tag.get_text().lower())
        if highlight_h2:
            parent_box = highlight_h2.find_parent("div")
            if parent_box:
                for col in parent_box.find_all("div", class_=re.compile(r"col-")):
                    icon = col.find("i", class_=re.compile(r"icon"))
                    text = col.get_text(strip=True).lower()
                    if icon:
                        classes = " ".join(icon.get("class", []))
                        is_active = ("green" in classes) and ("opacity" not in classes) and ("light" not in classes)
                    else:
                        is_active = False

                    for viet_name, key in self.AMENITY_MAPPING.items():
                        if viet_name in text:
                            detail[key] = 1 if is_active else 0

        # 2b. Combine checkboxes with NLP Regex matching on title + full_description
        combined_text = (title + " " + detail["full_description"]).lower()
        for k, pat in self.AMENITY_REGEX.items():
            if detail[k] == 0 and re.search(pat, combined_text):
                detail[k] = 1

        # 3. Confirmed post date from detail table if available
        for tr in soup.find_all("tr"):
            txt = tr.get_text().lower()
            if "ngày đăng" in txt:
                dt_match = parse_post_date(tr.get_text())
                if dt_match:
                    detail["confirmed_post_date"] = dt_match.isoformat()

        # 4. Clean breadcrumb category (strip trailing redundant title)
        breadcrumb = soup.find(["ol", "nav"], class_=re.compile(r"breadcrumb"))
        if breadcrumb:
            raw_cat = sanitize_text(breadcrumb.get_text(separator=" > ", strip=True))
            parts = [p.strip() for p in raw_cat.split(" > ")]
            if len(parts) > 1 and parts[-1].lower() == title.lower().strip():
                parts = parts[:-1]
            detail["category_name"] = " > ".join(parts)

        return detail

    def crawl_all(self):
        """
        Execute multi-endpoint crawler loop across all 24 districts & categories.
        Uses checkpoint resume, state persistence, domain constraint checks,
        and concurrent detail fetching for high throughput and zero data loss.
        """
        logger.info(f"=== Starting phongtro123 Crawler ===")
        logger.info(f"Target count: {self.target_count} items (Year >= {self.year_constraint})")

        total_collected = len(self.crawled_urls)
        if total_collected >= self.target_count:
            logger.info(f"Target already satisfied ({total_collected} items). Exiting.")
            return

        state_path = "data/raw/crawler_state.json"
        state = {}
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                state = {}

        # Build seed URL queue
        endpoints = []
        endpoints.append("https://phongtro123.com/tinh-thanh/ho-chi-minh")
        for dist in self.DISTRICT_ENDPOINTS:
            endpoints.append(f"https://phongtro123.com/tinh-thanh/ho-chi-minh/{dist}")
        for cat in self.CATEGORY_ENDPOINTS:
            endpoints.append(f"https://phongtro123.com/{cat}")

        saved_ep = state.get("endpoint")
        saved_page = state.get("page", 1)

        for ep_idx, endpoint in enumerate(endpoints):
            if len(self.crawled_urls) >= self.target_count:
                logger.info(f"Reached target {self.target_count} items. Stopping crawler.")
                break

            # Resume from saved page if on the same endpoint
            start_page = 1
            if saved_ep == endpoint and saved_page > 1:
                start_page = saved_page
            elif saved_ep and ep_idx < endpoints.index(saved_ep) if saved_ep in endpoints else False:
                # Skip previously completed endpoints
                continue

            logger.info(f"--- Crawling Endpoint [{ep_idx+1}/{len(endpoints)}]: {endpoint} (Starting at page {start_page}) ---")
            consecutive_old = 0
            page = start_page

            while page <= 100:  # Max 100 pages per endpoint
                if len(self.crawled_urls) >= self.target_count:
                    break

                page_url = f"{endpoint}?page={page}" if page > 1 else endpoint
                logger.info(f"Fetching page {page}: {page_url}")
                html = self.fetch_page(page_url)
                if not html:
                    logger.warning(f"Failed to fetch {page_url}, advancing page.")
                    page += 1
                    continue

                listings = self.parse_listing_items(html)
                if not listings:
                    logger.info(f"No listings found on page {page}, moving to next endpoint.")
                    break

                # 1. Filter candidates on current page
                candidates = []
                for item in listings:
                    if item["url"] in self.crawled_urls:
                        continue

                    if not self.is_valid_rental_listing(item):
                        post_year = item.get("post_year")
                        if post_year is not None and post_year < self.year_constraint:
                            consecutive_old += 1
                            if consecutive_old >= self.early_stop_count:
                                logger.info(
                                    f"Early stopping triggered! Encountered {consecutive_old} consecutive items with year < {self.year_constraint}. Moving to next endpoint."
                                )
                                break
                        continue
                    else:
                        consecutive_old = 0
                        candidates.append(item)

                if consecutive_old >= self.early_stop_count:
                    break

                if not candidates:
                    logger.info(f"Page {page}: No new valid candidates (all crawled or out-of-domain).")
                    state["endpoint"] = endpoint
                    state["page"] = page + 1
                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump(state, f)
                    page += 1
                    continue

                # 2. Fetch detail pages concurrently (3 workers) with polite delay
                def fetch_detail_worker(it):
                    time.sleep(random.uniform(0.5, 1.2))
                    d_html = self.fetch_page(it["url"])
                    d_data = self.parse_detail_page(d_html, title=it.get("title", "")) if d_html else {}
                    rec = {**it, **d_data}
                    rec["crawled_at"] = datetime.now().isoformat()
                    return rec

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    records = list(executor.map(fetch_detail_worker, candidates))

                for r in records:
                    self.scraped_records.append(r)
                    self.crawled_urls.add(r["url"])
                    if r.get("post_id"):
                        self.crawled_ids.add(str(r["post_id"]))

                if len(self.scraped_records) >= self.checkpoint_interval:
                    self.save_checkpoint()

                state["endpoint"] = endpoint
                state["page"] = page + 1
                state["total_collected"] = len(self.crawled_urls)
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f)

                logger.info(f"Page {page} finished. Added {len(records)} items. Total: {len(self.crawled_urls)}")
                page += 1

        # Final save
        self.save_checkpoint()
        logger.info(f"=== Crawling finished! Total records: {len(self.crawled_urls)} ===")


if __name__ == "__main__":
    crawler = PhongTro123Crawler()
    crawler.crawl_all()
