import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

def train():
    print("⏳ Loading Data...")
    # Check for files in current directory
    if not os.path.exists("train.csv") or not os.path.exists("Fertilizer Prediction.csv"):
        print("❌ Error: CSV files not found. Please ensure 'train.csv' and 'Fertilizer Prediction.csv' are in this folder.")
        return

    train_df = pd.read_csv("train.csv")
    original_df = pd.read_csv("Fertilizer Prediction.csv")

    # Standardize columns
    common_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 
                   'Potassium', 'Phosphorous', 'Soil Type', 'Crop Type', 'Fertilizer Name']

    # Combine datasets
    df = pd.concat([train_df[common_cols], original_df[common_cols]], axis=0).reset_index(drop=True)

    print("⚙️ Encoding Data...")
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fert = LabelEncoder()

    df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
    df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
    df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

    # --- Train Crop Model ---
    print("🚜 Training Crop Model...")
    X_crop = df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type']]
    y_crop = df['Crop Type']

    crop_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, enable_categorical=True)
    crop_model.fit(X_crop, y_crop)

    # --- Train Fertilizer Model ---
    print("💊 Training Fertilizer Model...")
    X_fert = df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type', 'Crop Type']]
    y_fert = df['Fertilizer Name']

    fert_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, enable_categorical=True)
    fert_model.fit(X_fert, y_fert)

    # --- SAVE EVERYTHING ---
    print("💾 Saving the brain to 'agri_brain.joblib'...")
    artifacts = {
        'crop_model': crop_model,
        'fert_model': fert_model,
        'le_soil': le_soil,
        'le_crop': le_crop,
        'le_fert': le_fert
    }
    joblib.dump(artifacts, 'agri_brain.joblib')
    print("✅ Done! You can now run 'inference.py'.")

if __name__ == "__main__":
    train()