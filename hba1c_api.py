"""
HbA1c Test Validation API
==========================
REST API for validating HbA1c test results and detecting blood disorders
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from hba1c_validation_model import (
    ClinicalDecisionSupport,
    generate_synthetic_training_data
)
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize the clinical decision support system
cds = ClinicalDecisionSupport()

# Train with synthetic data on startup (in production, use real data)
print("Initializing ML models...")
training_data = generate_synthetic_training_data(1000)
cds.initialize_models(training_data)
print("Models ready!")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'HbA1c Validation API',
        'models_loaded': True
    })


@app.route('/api/validate-hba1c', methods=['POST'])
def validate_hba1c():
    """
    Validate HbA1c test result for a patient
    
    Request body:
    {
        "patient_id": "string",
        "hba1c": float,
        "fasting_glucose": float,
        "random_glucose": float,
        "haemoglobin": float,
        "ferritin": float,
        ... (other blood parameters)
    }
    
    Returns:
    {
        "patient_id": "string",
        "test_validity": {...},
        "disorder_assessment": {...},
        "hba1c_values": {...},
        "clinical_recommendations": [...],
        "summary": "string"
    }
    """
    try:
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({
                'success': False,
                'error': 'No patient data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin']
        missing_fields = [field for field in required_fields if field not in patient_data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Perform comprehensive assessment
        result = cds.assess_test_result(patient_data)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'assessment': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/detect-anomaly', methods=['POST'])
def detect_anomaly():
    """
    Quick anomaly detection for HbA1c result
    
    Returns just the anomaly detection results
    """
    try:
        patient_data = request.get_json()
        
        anomaly_result = cds.anomaly_detector.detect_anomaly(patient_data)
        
        return jsonify({
            'success': True,
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'anomaly_detection': anomaly_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict-disorder', methods=['POST'])
def predict_disorder():
    """
    Predict blood disorder from lab parameters
    
    Request body:
    {
        "haemoglobin": float,
        "mcv": float,
        "ferritin": float,
        ... (blood parameters)
    }
    
    Returns:
    {
        "predicted_disorder": "string",
        "confidence": float,
        "all_probabilities": {...}
    }
    """
    try:
        patient_data = request.get_json()
        
        disorder_result = cds.disorder_classifier.predict_disorder(patient_data)
        
        return jsonify({
            'success': True,
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'disorder_prediction': disorder_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/correct-hba1c', methods=['POST'])
def correct_hba1c():
    """
    Get corrected HbA1c value accounting for blood disorders
    
    Request body:
    {
        "hba1c": float,
        "fasting_glucose": float,
        "haemoglobin": float,
        "disorder": "string" (optional)
        ... (other parameters)
    }
    
    Returns:
    {
        "measured_hba1c": float,
        "corrected_hba1c": float,
        "correction": float,
        "correction_percentage": float
    }
    """
    try:
        patient_data = request.get_json()
        
        correction_result = cds.hba1c_corrector.predict_corrected_hba1c(patient_data)
        
        return jsonify({
            'success': True,
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'correction': correction_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-validate', methods=['POST'])
def batch_validate():
    """
    Validate multiple HbA1c results in batch
    
    Request body:
    {
        "patients": [
            {...patient_data...},
            {...patient_data...}
        ]
    }
    """
    try:
        data = request.get_json()
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({
                'success': False,
                'error': 'No patient data provided'
            }), 400
        
        results = []
        for patient_data in patients:
            try:
                result = cds.assess_test_result(patient_data)
                results.append({
                    'success': True,
                    'assessment': result
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'error': str(e)
                })
        
        # Summary statistics
        unreliable_count = sum(
            1 for r in results 
            if r.get('success') and not r['assessment']['test_validity']['is_reliable']
        )
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'total_patients': len(patients),
            'processed': len(results),
            'unreliable_tests': unreliable_count,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    return jsonify({
        'success': True,
        'models': {
            'anomaly_detector': {
                'trained': cds.anomaly_detector.is_trained,
                'type': 'Isolation Forest',
                'purpose': 'Detect unreliable HbA1c results'
            },
            'disorder_classifier': {
                'trained': cds.disorder_classifier.is_trained,
                'type': 'Random Forest Classifier',
                'purpose': 'Classify blood disorders',
                'categories': ['none', 'iron_deficiency', 'thalassemia', 'sickle_cell', 'g6pd']
            },
            'hba1c_corrector': {
                'trained': cds.hba1c_corrector.is_trained,
                'type': 'Gradient Boosting Regressor',
                'purpose': 'Predict corrected HbA1c values'
            }
        },
        'training_data_size': 1000
    })


@app.route('/api/example-request', methods=['GET'])
def example_request():
    """Get example request format"""
    example = {
        "patient_id": "P12345",
        "hba1c": 7.2,
        "fasting_glucose": 120,
        "random_glucose": 140,
        "ogtt_2hr": 160,
        "avg_glucose_cgm": 125,
        "haemoglobin": 9.5,
        "rbc_count": 4.2,
        "mcv": 75,
        "mch": 25,
        "mchc": 32,
        "reticulocyte_count": 0.8,
        "wbc_count": 6.5,
        "platelet_count": 280,
        "serum_iron": 30,
        "ferritin": 12,
        "transferrin_saturation": 15,
        "tibc": 450,
        "bilirubin": 0.6,
        "ldh": 140,
        "haptoglobin": 100,
        "age": 35,
        "gender": "F",
        "disorder": "iron_deficiency",
        "rbc_lifespan_days": 90
    }
    
    return jsonify({
        'example_request': example,
        'required_fields': ['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin'],
        'optional_fields': list(set(example.keys()) - set(['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin']))
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("HbA1c Test Validation API Server")
    print("=" * 70)
    print("\nAvailable Endpoints:")
    print("  POST /api/validate-hba1c       - Full validation and assessment")
    print("  POST /api/detect-anomaly       - Quick anomaly detection")
    print("  POST /api/predict-disorder     - Blood disorder prediction")
    print("  POST /api/correct-hba1c        - Get corrected HbA1c value")
    print("  POST /api/batch-validate       - Validate multiple patients")
    print("  GET  /api/health               - Health check")
    print("  GET  /api/model-info           - Model information")
    print("  GET  /api/example-request      - Example request format")
    print("\nServer starting on http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
