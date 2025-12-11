
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/model_ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("model_data_ensemble")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_ensemble():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # 1. TEMİZLİK
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    print(f"✅ Analiz Verisi: {df.shape[0]} satır")
    
    # 2. FEATURE ENGINEERING
    y = np.log1p(df['jail'])
    
    # Yeni Özellikler Eklendi: county, case_type
    features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type' # YENİ
    ]
    prior_severity_cols = [c for c in df.columns if 'prior_charges_severity' in c]
    features.extend(prior_severity_cols)
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    
    # KATEGORİK İŞLEMLERİ
    cat_features = []
    
    # 1. Adım: Eksik Doldurma (Unknown)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna("Unknown").astype(str)
            cat_features.append(col)
        elif X[col].dtype.name == 'category':
             X[col] = X[col].astype(str).fillna("Unknown")
             cat_features.append(col)
             
    # Judge ID, County, Zip vb string olmalı
    for col in ['judge_id', 'county', 'zip']:
        if col in X.columns and col not in cat_features:
            X[col] = X[col].astype(str)
            cat_features.append(col)

    # 2. Adım: Label Encoding (XGBoost ve LightGBM için)
    # CatBoost kendi halleder ama diğerleri sayı ister.
    X_encoded = X.copy()
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        # Bilinmeyen değerleri handle etmek zordur LabelEncoder ile, o yüzden basit fit
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    print(f"📌 Kategorik Değişkenler: {cat_features}")

    # 3. K-FOLD STACKING
    # Veriyi bölmeden tüm veri üzerinde CV ile tahmin üretip metamodel eğitelim
    # (Gerçek bir projede hold-out test set ayrılmalı, burada R2 maksimizasyonu için CV yapıyoruz)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n🚀 MODEL 1: CatBoost Eğitiliyor (K-Fold)...")
    cat_model = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=8, cat_features=cat_features, verbose=0, random_seed=42
    )
    # CatBoost kategorik veriyi sever, X'i (encoded olmayan) veriyoruz
    cat_preds = cross_val_predict(cat_model, X, y, cv=kf, n_jobs=-1)
    
    print("🚀 MODEL 2: XGBoost Eğitiliyor (K-Fold)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42
    )
    # XGBoost encoded sever
    xgb_preds = cross_val_predict(xgb_model, X_encoded, y, cv=kf, n_jobs=-1)
    
    print("🚀 MODEL 3: LightGBM Eğitiliyor (K-Fold)...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=8, n_jobs=-1, random_state=42, verbose=-1
    )
    lgb_preds = cross_val_predict(lgb_model, X_encoded, y, cv=kf, n_jobs=-1)
    
    # 4. STACKING (EŞLEŞTİRME)
    print("\n🏗️ Stacking (Meta-Model) Eğitiliyor...")
    
    stacked_X = pd.DataFrame({
        'CatBoost': cat_preds,
        'XGBoost': xgb_preds,
        'LightGBM': lgb_preds
    })
    
    # Meta Model: Ridge Regression (Overfit olmasın diye)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stacked_X, y)
    
    final_preds_log = meta_model.predict(stacked_X)
    final_preds = np.expm1(final_preds_log)
    y_orig = np.expm1(y)
    
    # 5. DEĞERLENDİRME
    r2_log = r2_score(y, final_preds_log)
    r2_orig = r2_score(y_orig, final_preds)
    mae = mean_absolute_error(y_orig, final_preds)
    rmse = np.sqrt(mean_squared_error(y_orig, final_preds))
    
    print("\n📊 ENSEMBLE (STACKING) SONUÇLARI:")
    print(f"🔹 R2 Score (Log Scale): {r2_log:.4f}")
    print(f"🔹 R2 Score (Original): {r2_orig:.4f}")
    print(f"🔹 MAE: {mae:.2f} gün")
    print(f"🔹 RMSE: {rmse:.2f} gün")
    
    print("\n⚖️ Model Ağırlıkları (Hangi model ne kadar etkili?):")
    for name, coef in zip(stacked_X.columns, meta_model.coef_):
        print(f"  • {name}: {coef:.4f}")
        
    # Her bir modelin tekil başarısı
    print("\n🔍 Tekil Model Başarıları (Log R2):")
    print(f"  • CatBoost: {r2_score(y, cat_preds):.4f}")
    print(f"  • XGBoost: {r2_score(y, xgb_preds):.4f}")
    print(f"  • LightGBM: {r2_score(y, lgb_preds):.4f}")
    
    # Modeli Kaydet (Meta Model ve Base Modellerin Full Data ile Eğitilmesi Lazım üretim için)
    # Şimdilik analiz amaçlı skorları gösteriyoruz.
    
    if r2_log > 0.80:
        print("\n🎉 TEBRİKLER! %80 BARAJI AŞILDI!")
    else:
        print("\n⚠️ Hala %80 altındayız ama yaklaştık.")

if __name__ == "__main__":
    train_ensemble()
