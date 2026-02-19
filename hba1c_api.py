"""
HbA1c Validation API
====================
Flask API for validating HbA1c test results, detecting blood disorders,
and predicting corrected HbA1c values.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS so your HTML file can call the API

# Health check
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "HbA1c Validation API running"
    })


# HbA1c Validation Endpoint
@app.route("/api/validate-hba1c", methods=["POST"])
def validate_hba1c():
    data = request.get_json()

    # Extract values
    patient_id = data.get("patient_id")
    hba1c = data.get("hba1c")
    fasting_glucose = data.get("fasting_glucose")
    haemoglobin = data.get("haemoglobin")

    # Basic validation
    if not all([patient_id, hba1c, fasting_glucose, haemoglobin]):
        return jsonify({"error": "Missing required fields"}), 400

    # HbA1c classification
    if hba1c < 5.7:
        hba1c_status = "Normal"
    elif 5.7 <= hba1c < 6.5:
        hba1c_status = "Prediabetes"
    else:
        hba1c_status = "Diabetes"

    # Additional interpretation
    anemia_flag = haemoglobin < 10
    glucose_flag = fasting_glucose >= 126

    result = {
        "patient_id": patient_id,
        "hba1c": hba1c,
        "hba1c_status": hba1c_status,
        "fasting_glucose": fasting_glucose,
        "fasting_glucose_flag": "High" if glucose_flag else "Normal",
        "haemoglobin": haemoglobin,
        "anemia_flag": "Possible anemia" if anemia_flag else "Normal",
        "clinical_note": "Interpret HbA1c cautiously due to low haemoglobin"
            if anemia_flag else "HbA1c interpretation reliable"
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
# -----------------------------
# Run app
# -----------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
