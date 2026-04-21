# -*- coding: utf-8 -*-
import os
import sqlite3
import traceback
import numpy as np
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
# Ensure this matches your actual prediction file name
from phase3_predict_corrected_shimmed import predict_with_experts 

app = Flask(__name__)
CORS(app)

# ================= JWT CONFIG =================
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key-change-in-production-xyz')
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

init_db()  # Run once on startup

# ================= AUTH ROUTES =================
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not 
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
    if not 
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

# ================= ML ROUTES =================
TRAIN_MEAN = [0.12, -0.03, -0.15, -0.10, 0.05, 0.02, -0.08, 0.18, -0.04, -0.01, 0.12, 0.48]
TRAIN_STD = [0.75, 0.28, 0.85, 0.72, 1.15, 0.58, 0.38, 0.88, 0.48, 0.09, 0.58, 0.95]
MODEL_FEATURE_ORDER = [
    "soil_moisture", "nutrient_ec_dS_m", "npk_n_mgkg", "npk_p_mgkg", 
    "npk_k_mgkg", "soil_temp_c", "soil_ph", "air_temp_c", 
    "humidity_pct", "rainfall_forecast_mm", "crop_age_days", "plant_vi_proxy"
]

def standardize_input(raw_values):
    return [(0.0 if TRAIN_STD[i] == 0 else (val - TRAIN_MEAN[i]) / TRAIN_STD[i]) for i, val in enumerate(raw_values)]

def calculate_npk_rates(soil_n, soil_p, soil_k, crop_type="tomato", crop_age_days=60):
    crop_targets = {
        "tomato": {"vegetative": {"N": 50, "P": 20, "K": 200}, "flowering": {"N": 60, "P": 25, "K": 250}, "fruiting": {"N": 40, "P": 30, "K": 300}},
        "default": {"vegetative": {"N": 45, "P": 18, "K": 180}, "flowering": {"N": 55, "P": 22, "K": 220}, "fruiting": {"N": 35, "P": 25, "K": 250}}
    }
    stage = "vegetative" if crop_age_days < 45 else "flowering" if crop_age_days < 75 else "fruiting"
    targets = crop_targets.get(crop_type, crop_targets["default"])[stage]
    conversion = 1.4 * 30 * 0.1
    
    n_diff = (targets["N"] - soil_n) * conversion
    p_diff = (targets["P"] - soil_p) * conversion
    k_diff = (targets["K"] - soil_k) * conversion
    
    n_def, p_def, k_def = max(0, n_diff), max(0, p_diff), max(0, k_diff)
    n_ex, p_ex, k_ex = max(0, -n_diff), max(0, -p_diff), max(0, -k_diff)
    
    p2o5_req = (p_def * 2.29) if p_def > 0 else 0
    k2o_req = (k_def * 1.2) if k_def > 0 else 0
    dap_needed = p2o5_req / 0.46 if p2o5_req > 0 else 0
    n_from_dap = dap_needed * 0.18
    n_rem = max(0, (n_def / 0.46 if n_def > 0 else 0) - n_from_dap)
    urea_needed = n_rem / 0.46 if n_rem > 0 else 0
    mop_needed = k2o_req / 0.60 if k2o_req > 0 else 0

    return {
        "urea_kg_ha": round(urea_needed, 1), "dap_kg_ha": round(dap_needed, 1), "mop_kg_ha": round(mop_needed, 1),
        "total_n_kg_ha": round(n_def, 1), "total_p2o5_kg_ha": round(p2o5_req, 1), "total_k2o_kg_ha": round(k2o_req, 1),
        "needs_fertigation": (n_def > 15 or p_def > 10 or k_def > 60),
        "has_excess": (n_ex > 50 or p_ex > 20 or k_ex > 100),
        "deficiency_details": {
            "n_deficit_kg_ha": round(n_def, 1), "p_deficit_kg_ha": round(p_def, 1), "k_deficit_kg_ha": round(k_def, 1),
            "status": {
                "N": "DEFICIENT" if n_def > 15 else "EXCESS" if n_ex > 50 else "OPTIMAL",
                "P": "DEFICIENT" if p_def > 10 else "EXCESS" if p_ex > 20 else "OPTIMAL",
                "K": "DEFICIENT" if k_def > 60 else "EXCESS" if k_ex > 100 else "OPTIMAL"
            }
        }
    }

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.json
    if not payload: return jsonify({"error": "No JSON payload provided"}), 400

    raw_values = [float(payload.get(feat, 0)) for feat in MODEL_FEATURE_ORDER]
    std_payload = dict(zip(MODEL_FEATURE_ORDER, standardize_input(raw_values)))

    result = predict_with_experts(std_payload, need_threshold=0.45)
    try:
        result['fertilizer_breakdown'] = calculate_npk_rates(
            soil_n=payload.get('npk_n_mgkg', 0), soil_p=payload.get('npk_p_mgkg', 0), soil_k=payload.get('npk_k_mgkg', 0),
            crop_type=payload.get('crop_type', 'tomato'), crop_age_days=payload.get('crop_age_days', 60)
        )
    except Exception as e:
        print(f"NPK warning: {e}")

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
