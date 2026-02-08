import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import joblib

warnings.filterwarnings("ignore")

def train_and_predict():
    # --- 1. Load Data (Local Paths) ---
    print("⏳ Loading Data...")
    try:
        train = pd.read_csv("data/train.csv")
        original = pd.read_csv("data/Fertilizer Prediction.csv")
        test = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        print("❌ Error: CSV files not found. Make sure 'train.csv', 'Fertilizer Prediction.csv', and 'test.csv' are in the 'data' folder.")
        return

    # Standardize columns to ensure they match
    common_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 
                   'Potassium', 'Phosphorous', 'Soil Type', 'Crop Type', 'Fertilizer Name']

    # Combine datasets
    df = pd.concat([train[common_cols], original[common_cols]], axis=0).reset_index(drop=True)

    # --- 2. Encoders Setup ---
    print("⚙️ Encoding Data...")
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fert = LabelEncoder()

    df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
    df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
    df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

    # --- 3. TRAIN CROP MODEL ---
    print("🚜 Training Crop Model...")
    X_crop = df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type']]
    y_crop = df['Crop Type']

    crop_model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, 
        random_state=42, enable_categorical=True
    )
    crop_model.fit(X_crop, y_crop)

    # --- 4. TRAIN FERTILIZER MODEL ---
    print("💊 Training Fertilizer Model...")
    X_fert = df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type', 'Crop Type']]
    y_fert = df['Fertilizer Name']

    fert_model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, 
        random_state=42, enable_categorical=True
    )
    fert_model.fit(X_fert, y_fert)
    print("✅ Training Complete!\n")
    print("💾 Saving the model to a file...")

# We bundle everything into a dictionary so we only need one file
    artifacts = {
    'crop_model': crop_model,
    'fert_model': fert_model,
    'le_soil': le_soil,
    'le_crop': le_crop,
    'le_fert': le_fert
      }

# Save it to a file named 'agri_brain.joblib'
    joblib.dump(artifacts, 'agri_brain.joblib')

    print("✅ Model saved! You can now move 'agri_brain.joblib' to your API server.")

    # --- Helper Function inside the scope ---
    def recommend(temp, humid, moist, n, k, p, soil_type_name):
        try:
            # Check if soil type exists in encoder
            if soil_type_name not in le_soil.classes_:
                return f"Error: Soil '{soil_type_name}' not found. Valid: {list(le_soil.classes_)}", None, None

            soil_encoded = le_soil.transform([soil_type_name])[0]
            
            # Predict Crop
            input_data = pd.DataFrame([[temp, humid, moist, n, k, p, soil_encoded]], 
                                      columns=['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type'])
            
            crop_probs = crop_model.predict_proba(input_data)[0]
            top_3_idx = np.argsort(crop_probs)[-3:][::-1]
            top_3_crops = le_crop.inverse_transform(top_3_idx)
            top_3_scores = crop_probs[top_3_idx]
            
            suggestions = [(name, round(score * 100, 2)) for name, score in zip(top_3_crops, top_3_scores)]
            
            # Predict Fertilizer based on best crop
            input_data_fert = input_data.copy()
            input_data_fert['Crop Type'] = top_3_idx[0]
            
            pred_fert_idx = fert_model.predict(input_data_fert)[0]
            rec_fert = le_fert.inverse_transform([pred_fert_idx])[0]
            
            return suggestions, top_3_crops[0], rec_fert
            
        except Exception as e:
            return f"Error: {e}", None, None

    # --- 5. INTERACTIVE LOOP ---
    print("--- 🌿 AGRI RECOMMENDER SYSTEM 🌿 ---")
    print("Enter values to get a recommendation (or type 'exit' to quit).")
    
    while True:
        try:
            print("\nInput Data:")
            user_input = input("Type 'run' to use default test values, or press Enter to input manually: ").strip().lower()
            
            if user_input == 'exit':
                break
            
            if user_input == 'run':
                # Default Test Values
                t, h, m, n, k, p, s = 24, 60, 35, 12, 10, 35, 'Clayey'
            else:
                t = float(input("  Temperature: "))
                h = float(input("  Humidity: "))
                m = float(input("  Moisture: "))
                n = float(input("  Nitrogen: "))
                k = float(input("  Potassium: "))
                p = float(input("  Phosphorous: "))
                s = input("  Soil Type (Sandy/Loamy/Black/Red/Clayey): ").capitalize()

            suggestions, best_crop, fertilizer = recommend(t, h, m, n, k, p, s)

            if best_crop:
                print(f"\n📊 RESULTS for '{s}' Soil:")
                print("-" * 30)
                print("Top 3 Crops:")
                for rank, (crop, prob) in enumerate(suggestions, 1):
                    print(f"  {rank}. {crop:<10} ({prob}%)")
                print(f"\n🏆 Best Crop: {best_crop}")
                print(f"💊 Fertilizer: {fertilizer}")
                print("-" * 30)
            else:
                print(suggestions) # Print Error

        except ValueError:
            print("❌ Invalid input. Please enter numbers for measurements.")
    print("⏳ Loading model...")
artifacts = joblib.load('agri_brain.joblib')

# Extract the parts
crop_model = artifacts['crop_model']
fert_model = artifacts['fert_model']
le_soil = artifacts['le_soil']
le_crop = artifacts['le_crop']
le_fert = artifacts['le_fert']

print("✅ Model loaded and ready for API calls!")

# 2. THE PREDICTION FUNCTION (Called by API endpoint)
def get_prediction(temp, humid, moist, n, k, p, soil_type):
    try:
        # Encode Soil
        soil_encoded = le_soil.transform([soil_type])[0]
        
        # Predict Crop
        input_data = pd.DataFrame([[temp, humid, moist, n, k, p, soil_encoded]], 
                                  columns=['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type'])
        
        # Get Crop
        crop_probs = crop_model.predict_proba(input_data)[0]
        best_crop_idx = np.argsort(crop_probs)[-1] # Get #1 crop
        best_crop_name = le_crop.inverse_transform([best_crop_idx])[0]
        
        # Predict Fertilizer (using the predicted crop)
        input_data['Crop Type'] = best_crop_idx
        fert_idx = fert_model.predict(input_data)[0]
        fert_name = le_fert.inverse_transform([fert_idx])[0]
        
        return {"crop": best_crop_name, "fertilizer": fert_name}
        
    except Exception as e:
        return {"error": str(e)}        

if __name__ == "__main__":
    train_and_predict()