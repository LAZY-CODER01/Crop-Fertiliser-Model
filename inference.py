import joblib
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

# --- LOAD THE BRAIN ---
if not os.path.exists("agri_brain.joblib"):
    print("❌ Error: 'agri_brain.joblib' not found. Please run 'train_model.py' first!")
    exit()

print("⏳ Loading saved model...")
artifacts = joblib.load('agri_brain.joblib')

crop_model = artifacts['crop_model']
fert_model = artifacts['fert_model']
le_soil = artifacts['le_soil']
le_crop = artifacts['le_crop']
le_fert = artifacts['le_fert']

print("✅ Model loaded! Ready to predict.")

# Get valid options for display
valid_soils = list(le_soil.classes_)

def get_prediction(temp, humid, moist, n, k, p, soil_type):
    try:
        # 1. Check if Soil Type is valid
        # We try to find the soil in our list (ignoring case usually, but here we need exact match or close to it)
        if soil_type not in valid_soils:
            return {"Error": f"Unknown Soil '{soil_type}'.\nValid Options: {', '.join(valid_soils)}"}

        soil_encoded = le_soil.transform([soil_type])[0]
        
        # 2. Predict Crop
        input_data = pd.DataFrame([[temp, humid, moist, n, k, p, soil_encoded]], 
                                  columns=['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type'])
        
        crop_probs = crop_model.predict_proba(input_data)[0]
        top3_idx = np.argsort(crop_probs)[-3:][::-1]
        
        top3_names = le_crop.inverse_transform(top3_idx)
        top3_scores = crop_probs[top3_idx]
        
        crop_list = [f"{name} ({score*100:.1f}%)" for name, score in zip(top3_names, top3_scores)]
        
        # 3. Predict Fertilizer
        best_crop_idx = top3_idx[0]
        input_data['Crop Type'] = best_crop_idx
        
        fert_idx = fert_model.predict(input_data)[0]
        fert_name = le_fert.inverse_transform([fert_idx])[0]
        
        return {
            "Recommended Crops": crop_list,
            "Best Crop": top3_names[0],
            "Fertilizer": fert_name
        }

    except Exception as e:
        return {"Error": str(e)}

# --- INTERACTIVE LOOP ---
if __name__ == "__main__":
    print("\n--- 🌿 AGRI PREDICTOR 🌿 ---")
    print(f"ℹ️  Valid Soil Types: {', '.join(valid_soils)}")
    
    while True:
        try:
            val = input("\nPress Enter to test (or type 'exit'): ")
            if val.lower() == 'exit': break
            
            # Helper to get clean input
            t = float(input("Temp (e.g., 25): "))
            h = float(input("Humidity (e.g., 60): "))
            m = float(input("Moisture (e.g., 35): "))
            n = float(input("Nitrogen (e.g., 12): "))
            k = float(input("Potassium (e.g., 10): "))
            p = float(input("Phosphorous (e.g., 35): "))
            
            # Show options for soil
            print(f"Soil Options: {valid_soils}")
            s = input("Soil Type: ").strip() # Remove spaces
            
            result = get_prediction(t, h, m, n, k, p, s)
            
            print("\n-----------------------------")
            # CHECK FOR ERROR BEFORE PRINTING
            if "Error" in result:
                print(f"⚠️  {result['Error']}")
            else:
                print(f"🏆 Best Crop:  {result['Best Crop']}")
                print(f"💊 Fertilizer: {result['Fertilizer']}")
                print(f"📊 Alternatives: {', '.join(result['Recommended Crops'])}")
            print("-----------------------------")
            
        except ValueError:
            print("❌ Invalid input. Please enter numbers for T, H, M, N, K, P.")