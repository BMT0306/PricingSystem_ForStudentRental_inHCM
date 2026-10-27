"""
Module: src/explainer.py
Description: Explainable AI (XAI) using SHAP values. Generates Global summary beeswarm,
feature importance bar chart, sample waterfall plot, and residual diagnostics.
"""

import os
import logging
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.edgecolor"] = "#2d3748"


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ModelExplainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.output_dir = self.config["data_paths"]["reports_dir"]
        self.model_path = os.path.join(self.config["data_paths"]["models_dir"], "best_pipeline.joblib")
        os.makedirs(self.output_dir, exist_ok=True)

    def explain_model_and_save_plots(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Best pipeline not found at {self.model_path}. Please run trainer.py first.")
            return

        logger.info(f"Loading best pipeline from {self.model_path}")
        pipe = joblib.load(self.model_path)

        test_path = self.config["data_paths"]["test"]
        test_df = pd.read_csv(test_path, encoding="utf-8")

        features = [
            "area_m2", "dist_to_center_km", "dist_to_sgu_km",
            "dist_nearest_uni_km", "n_uni_within_2km",
            "dist_nearest_hospital_km", "n_hospitals_within_2km",
            "dist_nearest_metro_km", "amenity_score", "text_word_count",
            "has_ac", "has_mezzanine", "has_private_wc", "has_kitchen",
            "has_balcony_window", "has_fridge", "has_washer", "has_elevator",
            "is_fully_furnished", "no_curfew", "not_shared_owner",
            "has_security", "has_parking_basement", "is_all_inclusive",
            "female_only", "is_shared_dorm", "ward_new", "zone_group", "room_type"
        ]

        X_test = test_df[features]
        y_test = test_df["price_vnd"]

        # Residual Diagnostics
        y_pred = pipe.predict(X_test)
        self.plot_residual_diagnostics(y_test.values, y_pred)

        # Try SHAP
        try:
            import shap
            logger.info("Computing SHAP values on test set...")
            preprocessor = pipe.named_steps["preprocessor"]
            regressor = pipe.named_steps["regressor"].regressor_

            X_test_trans = preprocessor.transform(X_test)
            if hasattr(X_test_trans, "toarray"):
                X_test_trans = X_test_trans.toarray()

            explainer = shap.Explainer(regressor, X_test_trans)
            shap_values = explainer(X_test_trans[:300])

            # Beeswarm Plot
            plt.figure(figsize=(10, 8), dpi=300)
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
            plt.title("SHAP Global Summary Beeswarm Plot (Độ Đóng Góp Của Các Đặc Trưng)", fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "09_shap_summary_beeswarm.png"), bbox_inches="tight")
            plt.close()

            # Waterfall Sample Plot
            plt.figure(figsize=(10, 6), dpi=300)
            shap.plots.waterfall(shap_values[0], max_display=12, show=False)
            plt.title("SHAP Local Explanation (Giải Thích Đóng Góp Giá Cho Căn Phòng Mẫu)", fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "10_shap_waterfall_sample.png"), bbox_inches="tight")
            plt.close()

            logger.info("SHAP plots saved successfully!")
        except Exception as e:
            logger.warning(f"Could not compute SHAP plots: {e}")

    def plot_residual_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # 1. Actual vs Predicted
        axes[0].scatter(y_true / 1e6, y_pred / 1e6, alpha=0.4, color="#3182ce", s=25)
        lims = [0, 10.5]
        axes[0].plot(lims, lims, color="#e53e3e", linestyle="--", linewidth=1.5, label="Perfect Fit (y = ŷ)")
        axes[0].set_xlim(lims)
        axes[0].set_ylim(lims)
        axes[0].set_title("A. Giá Thực Tế vs. Giá Dự Đoán (Triệu VNĐ)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Giá thực tế (Triệu VNĐ/tháng)")
        axes[0].set_ylabel("Giá dự đoán (Triệu VNĐ/tháng)")
        axes[0].legend()

        # 2. Residual Distribution
        residuals = (y_pred - y_true) / 1e6
        sns.histplot(residuals, kde=True, ax=axes[1], color="#38a169", bins=30)
        axes[1].axvline(0, color="#2d3748", linestyle="--", linewidth=1.0)
        axes[1].set_title(f"B. Phân Phối Phần Dư (Residuals = ŷ - y)\nTrung vị Sai số: {np.median(residuals):+.2f}tr | MAE: {np.mean(np.abs(residuals)):.2f}tr", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Sai số dự đoán (Triệu VNĐ)")
        axes[1].set_ylabel("Số lượng mẫu")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "11_residual_diagnostics.png"), bbox_inches="tight")
        plt.close()
        logger.info("Residual diagnostics saved.")


if __name__ == "__main__":
    explainer = ModelExplainer()
    explainer.explain_model_and_save_plots()
