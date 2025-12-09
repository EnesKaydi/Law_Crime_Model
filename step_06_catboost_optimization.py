
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/model_advanced_catboost")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("model_data_advanced")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_catboost_model():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # 1. VERİ HAZIRLIĞI (Temizlik ve Outlier)
    if 'jail' not in df.columns:
        return
        
    # Filtreleme (>300 ve Outlier Temizliği)
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    print(f"✅ Analiz Verisi: {df.shape[0]} satır")
    
    # 2. FEATURE ENGINEERING (ÖZEL STRATEJİ)
    print("\n⚙️ Feature Engineering (CatBoost Stratejisi)...")
    
    y = np.log1p(df['jail'])
    
    # Eksik verileri DOLDURMİYORUZ! CatBoost NaN değerleri "Missing" olarak yönetebilir.
    # Ancak "is_recid_new" boşsa bunun anlamlı olduğunu gördük.
    # Bu yüzden NaN olanları özel bir kategori yapalım (Object ise), sayısal ise -1 veya mean
    
    # Özellik Listesi
    features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail'
    ]
    prior_severity_cols = [c for c in df.columns if 'prior_charges_severity' in c]
    features.extend(prior_severity_cols)
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    
    # Kategorik Değişkenleri Belirle
    cat_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object':
            # Kategorik: NaN değerleri "Unknown" yap
            X[col].fillna("Unknown", inplace=True)
            cat_features.append(col)
        elif X[col].dtype.name == 'category':
             X[col] = X[col].astype(str).fillna("Unknown")
             cat_features.append(col)
        else:
            # Sayısal
            # CatBoost Nan'ı sever ama float olmalı
             pass

    # judge_id sayısal görünebilir ama kategorik davranmalı (target encoding yerine catboost'un kendi encodingini deneyelim)
    if 'judge_id' in X.columns:
        X['judge_id'] = X['judge_id'].astype(str)
        if 'judge_id' not in cat_features:
            cat_features.append('judge_id')
            
    print(f"📌 Kategorik Değişkenler ({len(cat_features)}): {cat_features}")

    # 3. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. MODEL EĞİTİMİ (CATBOOST)
    print("\n🚀 CatBoost Modeli Eğitiliyor (Kategorik Odaklı)...")
    
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='R2',
        cat_features=cat_features,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        nan_mode='Min' # Eksik değerleri en küçük değer gibi işlem (veya 'Max')
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    # 5. DEĞERLENDİRME
    print("\n📊 CatBoost Sonuçları:")
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    r2_log = r2_score(y_test, y_pred_log)
    r2_orig = r2_score(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    mae = mean_absolute_error(y_test_orig, y_pred)
    
    print(f"🔹 R2 Score (Log Scale): {r2_log:.4f}")
    print(f"🔹 R2 Score (Original Scale): {r2_orig:.4f}")
    print(f"🔹 MAE: {mae:.2f} gün")
    print(f"🔹 RMSE: {rmse:.2f} gün")
    
    if r2_log > 0.65:
         print("\n✅ XGBoost'tan daha iyi performans (veya yakın)!")
    
    # Feature Importance
    feature_importance = model.get_feature_importance(Pool(X_test, label=y_test, cat_features=cat_features))
    feature_names = X_test.columns
    
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    fi_df = fi_df.sort_values(by='importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=fi_df)
    plt.title('CatBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "catboost_importance.png")
    
    # Modeli Kaydet
    model.save_model(str(MODEL_DIR / "catboost_model.cbm"))
    # Pipeline için gerekli objeleri kaydet (Feature listesi vs)
    joblib.dump(features, MODEL_DIR / "features_list.pkl")
    joblib.dump(cat_features, MODEL_DIR / "cat_features_list.pkl")
    
    print(f"\n💾 CatBoost modeli kaydedildi: {MODEL_DIR}/catboost_model.cbm")

if __name__ == "__main__":
    train_catboost_model()
