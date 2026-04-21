# -*- coding: utf-8 -*-
import os
import sqlite3
import traceback
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.optimize import differential_evolution

# Import your prediction module (ensure filename matches exactly)
from phase3_predict import predict_with_experts

app = Flask(__name__)
CORS(app)

# ================= JWT CONFIG =================
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'change-this-in-production-xyz123')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
jwt = JWTManager(app)

# ================= DATABASE SETUP =================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pfdss_users.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# ================= AUTH ROUTES =================
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    full_name = data.get('name', '').strip()

    if not email or not password or not full_name:
        return jsonify({"error": "Name, email, and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if '@' not in email:
        return jsonify({"error": "Invalid email format"}), 400

    password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('INSERT INTO users (email, password_hash, full_name) VALUES (?, ?, ?)',
                     (email, password_hash, full_name))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Email already registered"}), 409
    finally:
        conn.close()

    access_token = create_access_token(identity=email)
    return jsonify({
        "message": "Registration successful",
        "access_token": access_token,
        "user": {"email": email, "name": full_name}
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute('SELECT email, password_hash, full_name FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({"error": "Invalid email or password"}), 401

    access_token = create_access_token(identity=email)
    return jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "user": {"email": user['email'], "name": user['full_name']}
    }), 200

# ================= ML CONFIG =================
TRAIN_MEAN = [0.12, -0.03, -0.15, -0.10, 0.05, 0.02, -0.08, 0.18, -0.04, -0.01, 0.12, 0.48]
TRAIN_STD = [0.75, 0.28, 0.85, 0.72, 1.15, 0.58, 0.38, 0.88, 0.48, 0.09, 0.58, 0.95]

# FIXED: No trailing spaces in feature names
MODEL_FEATURE_ORDER = [
    "soil_moisture", "nutrient_ec_dS_m", "npk_n_mgkg", "npk_p_mgkg", 
    "npk_k_mgkg", "soil_temp_c", "soil_ph", "air_temp_c", 
    "humidity_pct", "rainfall_forecast_mm", "crop_age_days", "plant_vi_proxy"
]

def standardize_input(raw_values):
    standardized = []
    for i, val in enumerate(raw_values):
        if TRAIN_STD[i] == 0:
            standardized.append(0.0)
        else:
            standardized.append((val - TRAIN_MEAN[i]) / TRAIN_STD[i])
    return standardized

def calculate_npk_rates(soil_n, soil_p, soil_k, crop_type="tomato", crop_age_days=60):
    # FIXED: No trailing spaces in keys
    crop_targets = {
        "tomato": {
            "vegetative": {"N": 50, "P": 20, "K": 200},
            "flowering": {"N": 60, "P": 25, "K": 250},
            "fruiting": {"N": 40, "P": 30, "K": 300}
        },
        "default": {
            "vegetative": {"N": 45, "P": 18, "K": 180},
            "flowering": {"N": 55, "P": 22, "K": 220},
            "fruiting": {"N": 35, "P": 25, "K": 250}
        }
    }
    
    stage = "vegetative" if crop_age_days < 45 else "flowering" if crop_age_days < 75 else "fruiting"
    targets = crop_targets.get(crop_type, crop_targets["default"])[stage]

    # Nigerian standard conversion factor (0-30cm soil layer)
    bulk_density = 1.4
    root_depth = 0.3
    conversion_factor = bulk_density * root_depth * 10  # = 4.2

    n_diff = (targets["N"] - soil_n) * conversion_factor
    p_diff = (targets["P"] - soil_p) * conversion_factor
    k_diff = (targets["K"] - soil_k) * conversion_factor

    n_deficit = max(0, n_diff)
    p_deficit = max(0, p_diff)
    k_deficit = max(0, k_diff)
    n_excess = max(0, -n_diff)
    p_excess = max(0, -p_diff)
    k_excess = max(0, -k_diff)

    # Fertilizer calculations (Nigerian market standards)
    p2o5_required = (p_deficit * 2.29) if p_deficit > 0 else 0
    k2o_required = (k_deficit * 1.20) if k_deficit > 0 else 0
    
    dap_needed = p2o5_required / 0.46 if p2o5_required > 0 else 0
    n_from_dap = dap_needed * 0.18
    n_required = n_deficit / 0.46 if n_deficit > 0 else 0
    remaining_n = max(0, n_required - n_from_dap)
    urea_needed = remaining_n / 0.46 if remaining_n > 0 else 0
    mop_needed = k2o_required / 0.60 if k2o_required > 0 else 0

    return {
        "urea_kg_ha": round(urea_needed, 1),
        "dap_kg_ha": round(dap_needed, 1),
        "mop_kg_ha": round(mop_needed, 1),
        "total_n_kg_ha": round(n_deficit, 1),
        "total_p2o5_kg_ha": round(p2o5_required, 1),
        "total_k2o_kg_ha": round(k2o_required, 1),
        "needs_fertigation": (n_deficit > 15 or p_deficit > 10 or k_deficit > 60),
        "has_excess": (n_excess > 50 or p_excess > 20 or k_excess > 100),
        "deficiency_details": {
            "n_deficit_kg_ha": round(n_deficit, 1),
            "p_deficit_kg_ha": round(p_deficit, 1),
            "k_deficit_kg_ha": round(k_deficit, 1),
            "status": {
                "N": "DEFICIENT" if n_deficit > 15 else "EXCESS" if n_excess > 50 else "OPTIMAL",
                "P": "DEFICIENT" if p_deficit > 10 else "EXCESS" if p_excess > 20 else "OPTIMAL",
                "K": "DEFICIENT" if k_deficit > 60 else "EXCESS" if k_excess > 100 else "OPTIMAL"
            }
        }
    }

# ================= PREDICTION ROUTE =================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Extract and standardize inputs
        raw_values = [float(payload.get(feat, 0)) for feat in MODEL_FEATURE_ORDER]
        std_values = standardize_input(raw_values)
        std_payload = dict(zip(MODEL_FEATURE_ORDER, std_values))

        # Run ensemble prediction
        result = predict_with_experts(std_payload, need_threshold=0.45)

        # Add NPK breakdown
        try:
            npk_rates = calculate_npk_rates(
                soil_n=payload.get('npk_n_mgkg', 0),
                soil_p=payload.get('npk_p_mgkg', 0),
                soil_k=payload.get('npk_k_mgkg', 0),
                crop_type=payload.get('crop_type', 'tomato'),
                crop_age_days=payload.get('crop_age_days', 60)
            )
            result['fertilizer_breakdown'] = npk_rates
        except Exception as npk_error:
            print(f"⚠️ NPK calculation warning: {npk_error}")

        return jsonify(result)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ Prediction error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

# ================= HEALTH CHECK (for Render) =================
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "pfdss"}), 200

# ================= MAIN ENTRY POINT =================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
