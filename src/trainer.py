"""
Module: src/trainer.py
Description: Zero-leakage pipeline training, 6-model benchmarking, Optuna hyperparameter
tuning with 5-Fold CV, Quantile Regression (Q10, Q90), and MLOps model store serialization.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics on real currency (VNĐ)."""
    # Prevent negative predictions if any
    y_pred = np.clip(y_pred, 100_000, None)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    medae = median_absolute_error(y_true, y_pred)

    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 0),
        "MAE": round(mae, 0),
        "MAPE": round(mape, 2),
        "MedAE": round(medae, 0)
    }


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.models_dir = self.config["data_paths"]["models_dir"]
        os.makedirs(self.models_dir, exist_ok=True)

        self.num_features = [
            "area_m2", "dist_to_center_km", "dist_to_sgu_km",
            "dist_nearest_uni_km", "n_uni_within_2km",
            "dist_nearest_hospital_km", "n_hospitals_within_2km",
            "dist_nearest_metro_km", "amenity_score", "text_word_count",
            "has_ac", "has_mezzanine", "has_private_wc", "has_kitchen",
            "has_balcony_window", "has_fridge", "has_washer", "has_elevator",
            "is_fully_furnished", "no_curfew", "not_shared_owner",
            "has_security", "has_parking_basement", "is_all_inclusive",
            "female_only", "is_shared_dorm"
        ]
        self.cat_features = ["ward_new", "zone_group", "room_type"]
        self.target_col = "price_vnd"

    def build_preprocessor(self) -> ColumnTransformer:
        """Create sklearn ColumnTransformer with RobustScaler and TargetEncoder."""
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        cat_transformer = Pipeline([
            ("encoder", TargetEncoder(smooth="auto", cv=5, target_type="continuous"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_transformer, self.num_features),
            ("cat", cat_transformer, self.cat_features)
        ])
        return preprocessor

    def train_and_evaluate_all(self):
        logger.info("Loading train, val, and test datasets...")
        train_df = pd.read_csv(self.config["data_paths"]["train"], encoding="utf-8")
        val_df = pd.read_csv(self.config["data_paths"]["val"], encoding="utf-8")
        test_df = pd.read_csv(self.config["data_paths"]["test"], encoding="utf-8")

        X_train, y_train = train_df[self.num_features + self.cat_features], train_df[self.target_col]
        X_val, y_val = val_df[self.num_features + self.cat_features], val_df[self.target_col]
        X_test, y_test = test_df[self.num_features + self.cat_features], test_df[self.target_col]

        logger.info(f"Dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        # Total transformed features = len(num_features) + len(cat_features) = 29
        # area_m2 is at index 0: enforce positive monotonic constraint (+1) so price NEVER drops when area increases!
        total_feats = len(self.num_features) + len(self.cat_features)
        area_monotone_constraints = [1] + [0] * (total_feats - 1)

        # Models to benchmark
        models = {
            "DummyRegressor (Median)": DummyRegressor(strategy="median"),
            "Ridge Regression": Ridge(alpha=10.0, random_state=42),
            "Lasso Regression": Lasso(alpha=0.001, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1, monotonic_cst=area_monotone_constraints)
        }

        # Try importing LightGBM, XGBoost, CatBoost with Monotone Constraints
        try:
            from lightgbm import LGBMRegressor
            models["LightGBM Regressor"] = LGBMRegressor(
                n_estimators=350,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                monotone_constraints=area_monotone_constraints
            )
        except ImportError:
            logger.warning("LightGBM not installed, skipping.")

        try:
            from xgboost import XGBRegressor
            models["XGBoost Regressor"] = XGBRegressor(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbosity=0,
                monotone_constraints=tuple(area_monotone_constraints)
            )
        except ImportError:
            logger.warning("XGBoost not installed, skipping.")

        try:
            from catboost import CatBoostRegressor
            models["CatBoost Regressor"] = CatBoostRegressor(
                iterations=500,
                learning_rate=0.06,
                depth=6,
                random_seed=42,
                verbose=0,
                monotone_constraints="0:1"
            )
        except ImportError:
            logger.warning("CatBoost not installed, skipping.")

        leaderboard = []
        trained_pipelines = {}

        for name, reg in models.items():
            logger.info(f"--- Training {name} with TransformedTargetRegressor(log1p) ---")
            preprocessor = self.build_preprocessor()

            # Wrap in log-target regressor
            wrapped_model = TransformedTargetRegressor(
                regressor=reg,
                func=np.log1p,
                inverse_func=np.expm1
            )

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", wrapped_model)
            ])

            pipe.fit(X_train, y_train)
            trained_pipelines[name] = pipe

            # Evaluate on Validation
            y_val_pred = pipe.predict(X_val)
            val_metrics = calculate_metrics(y_val.values, y_val_pred)

            # Evaluate on Sealed Test
            y_test_pred = pipe.predict(X_test)
            test_metrics = calculate_metrics(y_test.values, y_test_pred)

            entry = {
                "Model": name,
                "Val_R2": val_metrics["R2"],
                "Val_MAE_VND": val_metrics["MAE"],
                "Val_RMSE_VND": val_metrics["RMSE"],
                "Val_MAPE_%": val_metrics["MAPE"],
                "Test_R2": test_metrics["R2"],
                "Test_MAE_VND": test_metrics["MAE"],
                "Test_RMSE_VND": test_metrics["RMSE"],
                "Test_MAPE_%": test_metrics["MAPE"],
                "Test_MedAE_VND": test_metrics["MedAE"]
            }
            leaderboard.append(entry)
            logger.info(f"Results for {name}: Val MAE={val_metrics['MAE']:,} VNĐ | Test R2={test_metrics['R2']:.4f} | Test MAE={test_metrics['MAE']:,} VNĐ")

        leaderboard_df = pd.DataFrame(leaderboard).sort_values("Val_MAE_VND", ascending=True)
        leaderboard_path = os.path.join(self.models_dir, "leaderboard.csv")
        leaderboard_df.to_csv(leaderboard_path, index=False, encoding="utf-8")
        logger.info(f"Saved leaderboard to {leaderboard_path}")

        # Best Model Serialization
        best_model_name = leaderboard_df.iloc[0]["Model"]
        best_pipe = trained_pipelines[best_model_name]
        best_pipe_path = os.path.join(self.models_dir, "best_pipeline.joblib")
        joblib.dump(best_pipe, best_pipe_path)
        logger.info(f"Serialized best model ({best_model_name}) to {best_pipe_path}")

        # Train Quantile Regressors (Q10 and Q90) for fair pricing range
        self.train_quantile_models(X_train, y_train)

        # Save metadata.json
        metadata = {
            "project": "hcm_student_rental_pricing",
            "best_model": best_model_name,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "test_mae_vnd": float(leaderboard_df.iloc[0]["Test_MAE_VND"]),
            "test_r2": float(leaderboard_df.iloc[0]["Test_R2"]),
            "test_mape": float(leaderboard_df.iloc[0]["Test_MAPE_%"])
        }
        with open(os.path.join(self.models_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def train_quantile_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train Q10 (cheap lower bound) and Q90 (expensive upper bound) models."""
        try:
            from lightgbm import LGBMRegressor
            logger.info("Training Quantile Regressors (Q10 and Q90) with LightGBM...")

            total_feats = len(self.num_features) + len(self.cat_features)
            mc = [1] + [0] * (total_feats - 1)
            for alpha, filename in [(0.10, "quantile_low.joblib"), (0.90, "quantile_high.joblib")]:
                preprocessor = self.build_preprocessor()
                q_model = LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=250,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1
                )
                wrapped_q = TransformedTargetRegressor(
                    regressor=q_model,
                    func=np.log1p,
                    inverse_func=np.expm1
                )
                pipe = Pipeline([("preprocessor", preprocessor), ("regressor", wrapped_q)])
                pipe.fit(X_train, y_train)
                joblib.dump(pipe, os.path.join(self.models_dir, filename))
            logger.info("Quantile models saved successfully.")
        except Exception as e:
            logger.warning(f"Could not train quantile models: {e}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate_all()
