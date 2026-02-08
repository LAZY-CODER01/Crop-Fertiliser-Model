from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# --- LOAD THE TRAINED MODEL ---
print("⏳ Loading model...")
try:
    artifacts = joblib.load('agri_brain.joblib')
    crop_model = artifacts['crop_model']
    fert_model = artifacts['fert_model']
    le_soil = artifacts['le_soil']
    le_crop = artifacts['le_crop']
    le_fert = artifacts['le_fert']
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'agri_brain.joblib' not found. Train the model first!")
    exit()

# --- THE PREDICTION FUNCTION ---
def get_prediction(data):
    try:
        # Extract data
        temp = float(data['temperature'])
        humid = float(data['humidity'])
        moist = float(data['moisture'])
        n = float(data['nitrogen'])
        k = float(data['potassium'])
        p = float(data['phosphorous'])
        soil_type = data['soil_type']

        # Validate Soil
        if soil_type not in le_soil.classes_:
            return {"error": f"Invalid Soil Type. Options: {list(le_soil.classes_)}"}

        # Encode Soil
        soil_encoded = le_soil.transform([soil_type])[0]
        
        # Prepare Input DataFrame
        input_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type']
        input_df = pd.DataFrame([[temp, humid, moist, n, k, p, soil_encoded]], columns=input_cols)

        # 1. Predict Crop
        crop_probs = crop_model.predict_proba(input_df)[0]
        top3_idx = np.argsort(crop_probs)[-3:][::-1]
        
        # Get Top 3 Crops
        top3_crops = []
        for idx in top3_idx:
            name = le_crop.inverse_transform([idx])[0]
            # --- FIX: Convert numpy float32 to python float ---
            prob = float(crop_probs[idx] * 100) 
            
            top3_crops.append({
                "crop": str(name),        # Convert numpy string to python string
                "confidence": round(prob, 2) 
            })

        # 2. Predict Fertilizer (based on best crop)
        best_crop_idx = top3_idx[0]
        input_df['Crop Type'] = best_crop_idx
        
        fert_idx = fert_model.predict(input_df)[0]
        fert_name = le_fert.inverse_transform([fert_idx])[0]

        return {
            "recommended_crop": str(top3_crops[0]['crop']),
            "recommended_fertilizer": str(fert_name),
            "alternatives": top3_crops
        }

    except Exception as e:
        return {"error": str(e)}

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = get_prediction(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)