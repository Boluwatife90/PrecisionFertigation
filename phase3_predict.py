import warnings
import sys
import types
import numpy as np
import pandas as pd
import joblib

# ==========================================
# CRITICAL SHIM FOR CyHalfSquaredError
# ==========================================
# Fixes "Can't get attribute 'CyHalfSquaredError' on <module '__main__'>"
class CyHalfSquaredError:
    """Dummy class to allow unpickling of older sklearn models."""
    def __init__(self, *args, **kwargs):
        pass

# 1. Patch __main__ (where the error specifically says it's looking)
import __main__
__main__.CyHalfSquaredError = CyHalfSquaredError

# 2. Patch _loss module
if '_loss' not in sys.modules:
    _loss_mod = types.ModuleType('_loss')
    _loss_mod.CyHalfSquaredError = CyHalfSquaredError
    sys.modules['_loss'] = _loss_mod
else:
    try: sys.modules['_loss'].CyHalfSquaredError = CyHalfSquaredError
    except: pass

# 3. Patch sklearn._loss (if it exists)
try:
    import sklearn._loss
    if not hasattr(sklearn._loss, 'CyHalfSquaredError'):
        sklearn._loss.CyHalfSquaredError = CyHalfSquaredError
except ImportError:
    pass
# ==========================================

def safe_joblib_load(path):
    """Load a joblib artifact robustly across minor version differences."""
    try:
        return joblib.load(path)
    except Exception as e:
        warnings.warn(f"joblib.load failed for {path}: {e}. Trying fallback...")
        try:
            return joblib.load(path, mmap_mode=None)
        except Exception:
            raise

def _median_fill(df_in, med=None):
    df2 = df_in.replace([np.inf, -np.inf], np.nan)
    if med is None:
        med2 = df2.median(numeric_only=True)
        return df2.fillna(med2).fillna(0.0), med2
    return df2.fillna(med).fillna(0.0), med

class Phase3StackedEnsemble:
    def __init__(self, phase2_base_bundle_path="phase2_base_models_bundle.joblib", 
                 phase3_meta_bundle_path="phase3_stacked_ensemble_bundle.joblib"):
        self.base_bundle = safe_joblib_load(phase2_base_bundle_path)
        self.meta_bundle = safe_joblib_load(phase3_meta_bundle_path)

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
        X_need = X_raw.reindex(columns=self.need_cols, fill_value=0.0)
        X_rate = X_raw.reindex(columns=self.rate_cols, fill_value=0.0)
        X_lag = X_raw.reindex(columns=self.lag_cols, fill_value=0.0)

        X_need_filled, _ = _median_fill(X_need)
        X_rate_filled, _ = _median_fill(X_rate)
        X_lag_filled, _ = _median_fill(X_lag)

        need_proba = self.need_model.predict_proba(X_need_filled)[:, 1]
        time_proba = self.time_model.predict_proba(X_need_filled)[:, 1]
        rate_pred = self.rate_model.predict(X_rate_filled)

        if self.ts_model is not None and len(self.lag_cols) > 0:
            try:
                moist_pred = self.ts_model.predict(X_lag_filled)
            except:
                moist_pred = np.repeat(np.nan, X_raw.shape[0])
        else:
            moist_pred = np.repeat(np.nan, X_raw.shape[0])

        meta_df = pd.DataFrame({
            "oof_need_proba": need_proba,
            "oof_rate_pred": rate_pred,
            "oof_moisture_pred": moist_pred,
            "oof_time_proba_48_72h": time_proba
        })

        for c in self.meta_raw_cols:
            if c in X_raw.columns:
                meta_df[c] = X_raw[c].values
            else:
                meta_df[c] = np.nan

        meta_df = meta_df.replace([np.inf, -np.inf], np.nan)
        meta_df = meta_df.fillna(meta_df.median(numeric_only=True))
        meta_df = meta_df.reindex(columns=self.meta_cols, fill_value=0.0)
        return meta_df.fillna(0.0)

    def predict(self, X_raw, need_threshold=0.5, clip_rate=(0.0, 200.0)):
        if not isinstance(X_raw, pd.DataFrame):
            X_raw = pd.DataFrame(X_raw)

        meta_df = self._build_meta_frame(X_raw)

        need_proba = self.meta_need.predict_proba(meta_df)[:, 1]
        # Handle binary vs multi-class output
        if need_proba.ndim > 1 and need_proba.shape[1] > 1:
             need_proba = need_proba[:, 1]
             
        need_label = (need_proba >= float(need_threshold)).astype(int)

        rate_pred = self.meta_rate.predict(meta_df)
        if clip_rate is not None:
            rate_pred = np.clip(rate_pred, float(clip_rate[0]), float(clip_rate[1]))

        timing_proba = self.meta_time.predict_proba(meta_df)[:, 1]

        return pd.DataFrame({
            "need_proba": need_proba,
            "need_label": need_label,
            "rate_pred": rate_pred,
            "timing_proba_48_72h": timing_proba
        })

def predict_with_experts(payload, base_bundle_path="phase2_base_models_bundle.joblib", 
                         ensemble_bundle_path="phase3_stacked_ensemble_bundle.joblib", 
                         need_threshold=0.45):
    model = Phase3StackedEnsemble(
        phase2_base_bundle_path=base_bundle_path,
        phase3_meta_bundle_path=ensemble_bundle_path
    )

    df_in = pd.DataFrame([payload])
    base_out = {}

    try:
        X_need = df_in.reindex(columns=model.need_cols, fill_value=0.0)
        X_need_filled, _ = _median_fill(X_need)
        proba = model.need_model.predict_proba(X_need_filled)
        if proba.shape[1] > 1:
            base_out["base_need_proba"] = float(proba[0, 1])
        else:
            base_out["base_need_proba"] = float(proba[0])
    except Exception:
        base_out["base_need_proba"] = 0.5

    try:
        X_rate = df_in.reindex(columns=model.rate_cols, fill_value=0.0)
        X_rate_filled, _ = _median_fill(X_rate)
        base_out["base_rate_raw"] = float(model.rate_model.predict(X_rate_filled)[0])
    except Exception:
        base_out["base_rate_raw"] = 0.0

    try:
        if model.ts_model is not None and len(model.lag_cols) > 0:
            X_lag = df_in.reindex(columns=model.lag_cols, fill_value=0.0)
            X_lag_filled, _ = _median_fill(X_lag)
            base_out["ts_pred_soil_moisture"] = float(model.ts_model.predict(X_lag_filled)[0])
        else:
            base_out["ts_pred_soil_moisture"] = None
    except Exception:
        base_out["ts_pred_soil_moisture"] = None

    try:
        ensemble_pred = model.predict(df_in, need_threshold=need_threshold).iloc[0].to_dict()
    except Exception:
        ensemble_pred = {
            "need_proba": base_out.get("base_need_proba", 0.5),
            "rate_pred": base_out.get("base_rate_raw", 0.0),
            "timing_proba_48_72h": 0.5
        }
        ensemble_pred["need_label"] = 1 if ensemble_pred["need_proba"] >= need_threshold else 0

    return {**ensemble_pred, "expert": {"base": base_out}}