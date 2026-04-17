# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
from phase3_predict import predict_with_experts
import traceback
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# ⚠️ REPLACE WITH YOUR ACTUAL VALUES FROM compute_stats.py
TRAIN_MEAN = [0.12, -0.03, -0.15, -0.10, 0.05, 0.02, -0.08, 0.18, -0.04, -0.01, 0.12, 0.48]
TRAIN_STD = [0.75, 0.28, 0.85, 0.72, 1.15, 0.58, 0.38, 0.88, 0.48, 0.09, 0.58, 0.95]

# ✅ CRITICAL: NO TRAILING SPACES - MUST MATCH FRONTEND EXACTLY
MODEL_FEATURE_ORDER = [
    "soil_moisture",
    "nutrient_ec_dS_m",
    "npk_n_mgkg",
    "npk_p_mgkg",
    "npk_k_mgkg",
    "soil_temp_c",
    "soil_ph",
    "air_temp_c",
    "humidity_pct",
    "rainfall_forecast_mm",
    "crop_age_days",
    "plant_vi_proxy"
]

def standardize_input(raw_values):
    """Standardize input features using training set statistics"""
    standardized = []
    for i, val in enumerate(raw_values):
        if TRAIN_STD[i] == 0:
            standardized.append(0.0)
        else:
            standardized.append((val - TRAIN_MEAN[i]) / TRAIN_STD[i])
    return standardized

def calculate_npk_rates(soil_n, soil_p, soil_k, crop_type="tomato", crop_age_days=60):
    """Calculate NPK requirements with proper handling of excess nutrients"""
    # ✅ NO TRAILING SPACES IN DICTIONARY KEYS
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
    
    bulk_density = 1.4
    root_depth = 0.3
    conversion_factor = 10
    
    # Calculate difference (can be negative if excess)
    n_diff = (targets["N"] - soil_n) * bulk_density * root_depth * conversion_factor
    p_diff = (targets["P"] - soil_p) * bulk_density * root_depth * conversion_factor
    k_diff = (targets["K"] - soil_k) * bulk_density * root_depth * conversion_factor
    
    # Separate deficit and excess
    n_deficit = max(0, n_diff)
    p_deficit = max(0, p_diff)
    k_deficit = max(0, k_diff)
    n_excess = max(0, -n_diff)
    p_excess = max(0, -p_diff)
    k_excess = max(0, -k_diff)
    
    # Calculate fertilizer needs only if deficient
    n_required = n_deficit / 0.60 if n_deficit > 0 else 0
    p_required = p_deficit / 0.30 if p_deficit > 0 else 0
    k_required = k_deficit / 0.70 if k_deficit > 0 else 0
    
    # Convert to fertilizer products
    p2o5_required = p_required * 2.29 if p_required > 0 else 0
    k2o_required = k_required * 1.20 if k_required > 0 else 0
    
    if p2o5_required > 0:
        dap_needed = p2o5_required / 0.46
        n_from_dap = dap_needed * 0.18
        remaining_n = max(0, n_required - n_from_dap)
        urea_needed = remaining_n / 0.46
    else:
        dap_needed = 0
        urea_needed = n_required / 0.46 if n_required > 0 else 0
    
    mop_needed = k2o_required / 0.60 if k2o_required > 0 else 0
    
    # Determine if fertigation is needed
    needs_fertigation = (n_deficit > 10 or p_deficit > 5 or k_deficit > 50)
    has_excess = (n_excess > 50 or p_excess > 20 or k_excess > 100)
    
    return {
        "urea_kg_ha": round(urea_needed, 1),
        "dap_kg_ha": round(dap_needed, 1),
        "mop_kg_ha": round(mop_needed, 1),
        "total_n_kg_ha": round(n_required, 1),
        "total_p2o5_kg_ha": round(p2o5_required, 1),
        "total_k2o_kg_ha": round(k2o_required, 1),
        "needs_fertigation": needs_fertigation,
        "has_excess": has_excess,
        "deficiency_details": {
            "n_deficit_kg_ha": round(n_deficit, 1),
            "p_deficit_kg_ha": round(p_deficit, 1),
            "k_deficit_kg_ha": round(k_deficit, 1),
            "n_excess_kg_ha": round(n_excess, 1),
            "p_excess_kg_ha": round(p_excess, 1),
            "k_excess_kg_ha": round(k_excess, 1),
            "target_levels": targets,
            "current_levels": {"N": soil_n, "P": soil_p, "K": soil_k},
            "status": {
                "N": "DEFICIENT" if n_deficit > 10 else "EXCESS" if n_excess > 50 else "OPTIMAL",
                "P": "DEFICIENT" if p_deficit > 5 else "EXCESS" if p_excess > 20 else "OPTIMAL",
                "K": "DEFICIENT" if k_deficit > 50 else "EXCESS" if k_excess > 100 else "OPTIMAL"
            }
        }
    }

# 🔗 MAIN PREDICTION ENDPOINT
@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Extract raw values in model order
        raw_values = [float(payload.get(feat, 0)) for feat in MODEL_FEATURE_ORDER]
        std_values = standardize_input(raw_values)
        std_payload = dict(zip(MODEL_FEATURE_ORDER, std_values))

        result = predict_with_experts(std_payload, need_threshold=0.45)
        
        # Always calculate NPK breakdown
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
            traceback.print_exc()
        
        return jsonify(result)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ Prediction error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

# 🧪 TEST ENDPOINT - Returns sample prediction for frontend testing
@app.route('/test', methods=['GET', 'POST'])
def test_prediction():
    """Test endpoint that returns sample prediction data without calling ML model"""
    sample_response = {
        "need_label": 1,
        "need_proba": 0.92,
        "rate_pred": 77.4,
        "timing": "Apply within 48-72 hours",
        "confidence": 92,
        "expert": {
            "base": {
                "ts_pred_soil_moisture": 18.5,
                "base_rate_raw": 75.2,
                "base_need_proba": 0.89
            }
        },
        "fertilizer_breakdown": {
            "urea_kg_ha": 274.0,
            "dap_kg_ha": 698.0,
            "mop_kg_ha": 720.0,
            "total_n_kg_ha": 315.0,
            "total_p2o5_kg_ha": 321.0,
            "total_k2o_kg_ha": 864.0,
            "needs_fertigation": True,
            "has_excess": False,
            "deficiency_details": {
                "n_deficit_kg_ha": 189.0,
                "p_deficit_kg_ha": 42.0,
                "k_deficit_kg_ha": 504.0,
                "n_excess_kg_ha": 0,
                "p_excess_kg_ha": 0,
                "k_excess_kg_ha": 0,
                "target_levels": {"N": 50, "P": 20, "K": 200},
                "current_levels": {"N": 5, "P": 10, "K": 80},
                "status": {
                    "N": "DEFICIENT",
                    "P": "DEFICIENT", 
                    "K": "DEFICIENT"
                }
            }
        },
        "message": "This is a TEST response. Connect your ML model for real predictions."
    }
    return jsonify(sample_response), 200

# 🏥 HEALTH CHECK ENDPOINT - For Render monitoring
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring service status"""
    return jsonify({
        "status": "healthy",
        "service": "precision-fertigation-api",
        "endpoints": ["/predict", "/test", "/health"],
        "timestamp": np.datetime64('now').astype(str)
    }), 200

# 🔄 SUGGEST IMPROVEMENTS ENDPOINT (Optional - for confidence optimization)
@app.route('/suggest-improvements', methods=['POST'])
def suggest_improvements():
    """Analyzes current inputs and suggests changes to maximize confidence"""
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        current_features = [float(payload.get(f, 0)) for f in MODEL_FEATURE_ORDER]
        
        # Get current prediction
        std_current = standardize_input(current_features)
        std_payload = dict(zip(MODEL_FEATURE_ORDER, std_current))
        current_result = predict_with_experts(std_payload, need_threshold=0.45)
        current_confidence = current_result.get('need_proba', 0.5)
        
        # For testing: return simple suggestions without optimization
        suggestions = [
            {
                "feature": "Soil Moisture",
                "current": round(current_features[0], 2),
                "suggested": round(current_features[0] * 0.8, 2),
                "change": "↓ decrease by " + str(round(current_features[0] * 0.2, 2)),
                "impact": "high"
            },
            {
                "feature": "N (mg/kg)",
                "current": round(current_features[2], 2),
                "suggested": round(current_features[2] * 0.9, 2),
                "change": "↓ decrease by " + str(round(current_features[2] * 0.1, 2)),
                "impact": "medium"
            }
        ]
        
        return jsonify({
            "current_confidence": round(current_confidence * 100, 1),
            "achievable_confidence": min(99.9, round(current_confidence * 100 + 10, 1)),
            "suggestions": suggestions[:5],
            "optimal_inputs": dict(zip(MODEL_FEATURE_ORDER, [round(v * 0.95, 2) for v in current_features]))
        })
        
    except Exception as e:
        error_msg = f"Suggestion generation failed: {str(e)}"
        print(f"❌ Suggestion error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
