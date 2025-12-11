
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
from pathlib import Path

# Yollar
MODEL_DIR = Path("../model_data_advanced")
MODEL_PATH = MODEL_DIR / "catboost_model.cbm"
FEATURES_PATH = MODEL_DIR / "features_list.pkl"
CAT_FEATURES_PATH = MODEL_DIR / "cat_features_list.pkl"

def load_system():
    print("⏳ Sistem Yükleniyor...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model dosyası bulunamadı! Önce eğitimi çalıştırın.")
        
    # Modeli Yükle
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    
    # Feature Listesiniz Yükle
    feature_names = joblib.load(FEATURES_PATH)
    cat_features = joblib.load(CAT_FEATURES_PATH)
    
    print("✅ Model ve Sistem Hazır!")
    return model, feature_names, cat_features

def predict_sentence(model, feature_names, cat_features, input_data):
    """
    input_data: Sözlük (Dictionary) formatında kullanıcı verisi
    """
    # 1. Veriyi DataFrame'e Çevir
    data_row = {}
    for feature in feature_names:
        val = input_data.get(feature)
        # Eksikse NaN
        if val is None:
            data_row[feature] = np.nan
        else:
            data_row[feature] = val
            
    df = pd.DataFrame([data_row])
    
    # 2. Veri Tipi Dönüşümleri (Kategorik Zorlama)
    # Eğitimdeki kategorik kolonları BİLİYORUZ, onları zorla düzeltelim
    for col in cat_features:
        if col in df.columns:
            # Önce bilinmeyenleri Unknown yap
            df[col] = df[col].fillna("Unknown")
            # Sonra string yap
            df[col] = df[col].astype(str)
            # 'nan' stringi varsa temizle
            df.loc[df[col] == 'nan', col] = "Unknown"
            
    # 3. Tahmin Yap
    prediction_log = model.predict(df)[0]
    prediction_days = np.expm1(prediction_log)
    
    return prediction_days

# --- ÖRNEK SENARYO (WEB'DEN GELEN VERİ) ---
if __name__ == "__main__":
    model, feature_names, cat_features = load_system()
    
    # SENARYO 1: Sert Bir Hakim, Ciddi Bir Suç
    kullanici_verisi_1 = {
        'highest_severity': 10,
        'violent_crime': 1,
        'is_recid_new': 1,
        'sex': 'Male',
        'age_offense': 30,
        'judge_id': '673',
        'year': 2024
    }
    
    # SENARYO 2: Yumuşak Hakim, İlk Suç
    kullanici_verisi_2 = {
        'highest_severity': 10,
        'violent_crime': 1,
        'is_recid_new': 0,
        'sex': 'Male',
        'age_offense': 30,
        'judge_id': '221',
        'year': 2024
    }
    
    print("\n--- SENARYO 1: Sert Hakim + Sabıkalı ---")
    ceza1 = predict_sentence(model, feature_names, cat_features, kullanici_verisi_1)
    print(f"⚖️  Tahmin Edilen Ceza: {ceza1:.0f} GÜN ({ceza1/365:.1f} YIL)")
    
    print("\n--- SENARYO 2: Yumuşak Hakim + İlk Suç ---")
    ceza2 = predict_sentence(model, feature_names, cat_features, kullanici_verisi_2)
    print(f"⚖️  Tahmin Edilen Ceza: {ceza2:.0f} GÜN ({ceza2/365:.1f} YIL)")
    
    fark = ceza1 - ceza2
    print(f"\n📉 Fark: {fark:.0f} Gün! (Modelin hakim ve sabıka duyarlılığı)")
