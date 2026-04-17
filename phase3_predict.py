# phase3_predict.py
# Deployable helper for Phase 3 stacked ensemble
# Loads Phase 2 base bundle + Phase 3 meta bundle and produces unified outputs.

DEFAULT_BASE_BUNDLE = 'phase2_base_models_bundle.joblib'
DEFAULT_ENSEMBLE_BUNDLE = 'phase3_stacked_ensemble_bundle.joblib'

import numpy as np
import pandas as pd
import joblib


def _median_fill(df_in, med=None):
    df2 = df_in.replace([np.inf, -np.inf], np.nan)
    if med is None:
        med2 = df2.median(numeric_only=True)
        return df2.fillna(med2).fillna(0.0), med2
    return df2.fillna(med).fillna(0.0), med


class Phase3StackedEnsemble:
    def __init__(self, phase2_base_bundle_path="phase2_base_models_bundle.joblib", phase3_meta_bundle_path="phase3_stacked_ensemble_bundle.joblib"):
        self.base_bundle = joblib.load(phase2_base_bundle_path)
        self.meta_bundle = joblib.load(phase3_meta_bundle_path)

        self.need_model = self.base_bundle["models"]["need_classifier"]
        self.rate_model = self.base_bundle["models"]["rate_regressor"]
        self.ts_model = self.base_bundle["models"]["ts_moisture_forecaster"]
        self.time_model = self.base_bundle["models"]["timing_classifier_48_72h_bin"]

        self.need_cols = self.base_bundle["feature_sets"]["need_feature_cols"]
        self.rate_cols = self.base_bundle["feature_sets"]["rate_feature_cols"]
        self.lag_cols = self.base_bundle["feature_sets"]["lag_feature_cols"]

        self.meta_need = self.meta_bundle["models"]["need_meta"]
        self.meta_rate = self.meta_bundle["models"]["rate_meta"]
        self.meta_time = self.meta_bundle["models"]["timing_meta"]

        self.meta_cols = self.meta_bundle["meta_features"]["final_meta_feature_list"]
        self.meta_base_cols = self.meta_bundle["meta_features"]["base_oof_outputs"]
        self.meta_raw_cols = self.meta_bundle["meta_features"]["raw_features_added"]

    def _build_meta_frame(self, X_raw):
        # Base outputs
        X_need = X_raw.reindex(columns=self.need_cols, fill_value=0.0)
        X_rate = X_raw.reindex(columns=self.rate_cols, fill_value=0.0)
        X_lag = X_raw.reindex(columns=self.lag_cols, fill_value=0.0)

        X_need_filled, _ = _median_fill(X_need)
        X_rate_filled, _ = _median_fill(X_rate)
        X_lag_filled, _ = _median_fill(X_lag)

        need_proba = self.need_model.predict_proba(X_need_filled)[:, 1]
        time_proba = self.time_model.predict_proba(X_need_filled)[:, 1]
        rate_pred = self.rate_model.predict(X_rate_filled)

        # Moisture forecast requires lag cols; if none, set nan
        if len(self.lag_cols) > 0:
            moist_pred = self.ts_model.predict(X_lag_filled)
        else:
            moist_pred = np.repeat(np.nan, X_raw.shape[0])

        meta_df = pd.DataFrame({
            "oof_need_proba": need_proba,
            "oof_rate_pred": rate_pred,
            "oof_moisture_pred": moist_pred,
            "oof_time_proba_48_72h": time_proba
        })

        # Raw anchors
        for c in self.meta_raw_cols:
            if c in X_raw.columns:
                meta_df[c] = X_raw[c].values
            else:
                meta_df[c] = np.nan

        meta_df = meta_df.replace([np.inf, -np.inf], np.nan)
        meta_df = meta_df.fillna(meta_df.median(numeric_only=True))
        meta_df = meta_df.reindex(columns=self.meta_cols, fill_value=0.0)
        meta_df = meta_df.fillna(0.0)
        return meta_df

    def predict(self, X_raw, need_threshold=0.5, clip_rate=(0.0, 200.0)):
        if not isinstance(X_raw, pd.DataFrame):
            X_raw = pd.DataFrame(X_raw)

        meta_df = self._build_meta_frame(X_raw)

        need_proba = self.meta_need.predict_proba(meta_df)[:, 1]
        need_label = (need_proba >= float(need_threshold)).astype(int)

        rate_pred = self.meta_rate.predict(meta_df)
        if clip_rate is not None:
            rate_pred = np.clip(rate_pred, float(clip_rate[0]), float(clip_rate[1]))

        timing_proba = self.meta_time.predict_proba(meta_df)[:, 1]

        out = pd.DataFrame({
            "need_proba": need_proba,
            "need_label": need_label,
            "rate_pred": rate_pred,
            "timing_proba_48_72h": timing_proba
        })
        return out


def predict_with_experts(payload, base_bundle_path=DEFAULT_BASE_BUNDLE, ensemble_bundle_path=DEFAULT_ENSEMBLE_BUNDLE, need_threshold=0.45):
    """Return ensemble outputs plus expert/base-model outputs.

    Args:
        payload: dict of sensor inputs
        need_threshold: decision threshold (default 0.45 for higher sensitivity)
    """
    # Load bundles
    base_bundle = joblib.load(base_bundle_path)
    ens_bundle = joblib.load(ensemble_bundle_path)

    # Create model instance
    model = Phase3StackedEnsemble(
        phase2_base_bundle_path=base_bundle_path,
        phase3_meta_bundle_path=ensemble_bundle_path
    )

    # Convert payload dict to DataFrame
    df_in = pd.DataFrame([payload])

    # === Get base model predictions manually ===
    base_out = {}

    # Need classifier
    try:
        X_need = df_in.reindex(columns=model.need_cols, fill_value=0.0)
        X_need_filled, _ = _median_fill(X_need)
        base_need_proba = float(model.need_model.predict_proba(X_need_filled)[0, 1])
        base_out["base_need_proba"] = base_need_proba
        print(f"DEBUG: Base classifier output = {base_need_proba:.3f}")  # ADDED DEBUG
    except Exception:
        base_out["base_need_proba"] = None

    # Rate regressor
    try:
        X_rate = df_in.reindex(columns=model.rate_cols, fill_value=0.0)
        X_rate_filled, _ = _median_fill(X_rate)
        base_rate_raw = float(model.rate_model.predict(X_rate_filled)[0])
        base_out["base_rate_raw"] = base_rate_raw
        print(f"DEBUG: Base regressor output = {base_rate_raw:.1f} kg/ha")  # ADDED DEBUG
    except Exception:
        base_out["base_rate_raw"] = None

    # Time-series model
    try:
        if model.ts_model is not None and len(model.lag_cols) > 0:
            X_lag = df_in.reindex(columns=model.lag_cols, fill_value=0.0)
            X_lag_filled, _ = _median_fill(X_lag)
            ts_pred = float(model.ts_model.predict(X_lag_filled)[0])
            base_out["ts_pred_soil_moisture"] = ts_pred
            print(f"DEBUG: Time-series forecast = {ts_pred:.1f}%")  # ADDED DEBUG
        else:
            base_out["ts_pred_soil_moisture"] = None
    except Exception:
        base_out["ts_pred_soil_moisture"] = None

    # === Get ensemble prediction WITH CUSTOM THRESHOLD ===
    ensemble_pred = model.predict(df_in, need_threshold=need_threshold).iloc[0].to_dict()
    print(f"DEBUG: Meta-learner final output = {ensemble_pred['need_proba']:.3f}")  # ADDED DEBUG

    return {
        **ensemble_pred,
        "expert": {
            "base": base_out
        }
    }