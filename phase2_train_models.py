import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, roc_auc_score, accuracy_score
)
import joblib
import os

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load data
df = pd.read_csv("css'.csv", header=None)

# Assign column names (based on your data structure)
feature_names = [
    'soil_moisture', 'EC', 'N', 'P', 'K', 'soil_temp', 'pH',
    'air_temp', 'humidity', 'rainfall', 'ndvi_proxy', 'growth_stage_encoded'
]

target_names = [
    'fertigation_need', 'fertigation_need_binary',
    'fertigation_rate', 'fertigation_timing',
    'fertilizer_type', 'video_url', 'days_after_planting'
]

# Separate features and targets
X = df.iloc[:, :12].values  # 12 input features
y_need = df.iloc[:, 12].values  # 'low', 'medium', 'high'
y_need_bin = df.iloc[:, 13].values.astype(int)  # 0/1
y_rate = df.iloc[:, 14].values  # continuous
y_timing = df.iloc[:, 15].values  # e.g., '48_72h'

# Encode categorical targets
le_need = LabelEncoder()
y_need_enc = le_need.fit_transform(y_need)  # 0,1,2

le_timing = LabelEncoder()
y_timing_enc = le_timing.fit_transform(y_timing)

# Split data (70/15/15) with stratification on need
X_temp, X_test, y_need_temp, y_need_test, y_need_bin_temp, y_need_bin_test, \
y_rate_temp, y_rate_test, y_timing_temp, y_timing_test = train_test_split(
    X, y_need_enc, y_need_bin, y_rate, y_timing_enc,
    test_size=0.15, stratify=y_need_enc, random_state=42
)

X_train, X_val, y_need_train, y_need_val, y_need_bin_train, y_need_bin_val, \
y_rate_train, y_rate_val, y_timing_train, y_timing_val = train_test_split(
    X_temp, y_need_temp, y_need_bin_temp, y_rate_temp, y_timing_temp,
    test_size=0.1765, stratify=y_need_temp, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ==============================
# TRAIN MODELS
# ==============================

# 1. Tree-Based Regressor (fertigation_rate)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_rate_train)
y_rate_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_rate_test, y_rate_pred)
rmse = np.sqrt(mean_squared_error(y_rate_test, y_rate_pred))
r2 = r2_score(y_rate_test, y_rate_pred)

# 2. Classifier (fertigation_need: low/med/high)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_need_train)
y_need_pred = classifier.predict(X_test)
y_need_proba = classifier.predict_proba(X_test)
f1_need = f1_score(y_need_test, y_need_pred, average='macro')
auc_need = roc_auc_score(y_need_test, y_need_proba, multi_class='ovr')

# 3. Timing Classifier (fertigation_timing)
timing_model = RandomForestClassifier(n_estimators=100, random_state=42)
timing_model.fit(X_train, y_timing_train)
y_timing_pred = timing_model.predict(X_test)
acc_timing = accuracy_score(y_timing_test, y_timing_pred)
f1_timing = f1_score(y_timing_test, y_timing_pred, average='macro')

# ==============================
# SAVE RESULTS
# ==============================

# Metrics report
metrics = {
    "fertigation_rate": {"MAE": mae, "RMSE": rmse, "R2": r2},
    "fertigation_need": {"F1_macro": f1_need, "ROC_AUC_OvR": auc_need},
    "fertigation_timing": {"Accuracy": acc_timing, "F1_macro": f1_timing}
}

# Save metrics
import json
with open("results/phase2_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save models
joblib.dump(regressor, "models/regressor_rate.pkl")
joblib.dump(classifier, "models/classifier_need.pkl")
joblib.dump(timing_model, "models/classifier_timing.pkl")

# Save encoders (for later use)
joblib.dump(le_need, "models/label_encoder_need.pkl")
joblib.dump(le_timing, "models/label_encoder_timing.pkl")

# Save train/val/test indices (for Phase 3 OOF)
np.save("results/train_indices.npy", X_train)  # or save actual indices if needed

print("\n✅ PHASE 2 COMPLETE")
print("Saved to:")
print("  - models/")
print("  - results/phase2_metrics.json")
print("\nMetrics:")
print(json.dumps(metrics, indent=2))