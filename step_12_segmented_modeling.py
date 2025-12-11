
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/segmented_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("model_data_segmented")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_segmented_models():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # Özellikler (Mevcut en iyi set + yeni özellikler)
    features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    available_features = [f for f in features if f in df.columns]
    
    # Kategorik Belirleme
    cat_features = []
    X = df[available_features].copy()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("Unknown").astype(str)
            cat_features.append(col)
            
    # Judge ID, Zip, County zorla string
    for col in ['judge_id', 'county', 'zip']:
        if col in X.columns:
            X[col] = X[col].astype(str)
            if col not in cat_features: cat_features.append(col)

    print(f"📌 Kategorik Değişkenler: {cat_features}")

    # --- SEGMENTASYON STRATEJİSİ: THRESHOLD = 3000 GÜN ---
    # Bilimsel Yaklaşım: Heteroskedastisiteyi (değişen varyans) azaltmak için veriyi homojen kümelere ayırıyoruz.
    THRESHOLD = 3000
    
    mask_low = df['jail'] <= THRESHOLD
    mask_high = df['jail'] > THRESHOLD
    
    df_low = df[mask_low].copy()
    df_high = df[mask_high].copy()
    
    print(f"\n📊 Segmentasyon Özeti:")
    print(f"  • Model 1 (Mainstream / 300-{THRESHOLD} gün): {len(df_low)} veri (%{len(df_low)/len(df)*100:.1f})")
    print(f"  • Model 2 (High Severity / {THRESHOLD}+ gün): {len(df_high)} veri (%{len(df_high)/len(df)*100:.1f})")
    
    # --- MODEL 1: MAINSTREAM (Düşük/Orta Ceza) ---
    print("\n🚀 MODEL 1: Mainstream Eğitiliyor...")
    y_low = np.log1p(df_low['jail'])
    X_low = df_low[available_features]
    
    # NaN Handle (CatBoost için)
    X_low_processed = X_low.copy()
    for col in cat_features:
        X_low_processed[col] = X_low_processed[col].fillna("Unknown").astype(str)

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_low_processed, y_low, test_size=0.2, random_state=42)
    
    model_low = CatBoostRegressor(
        iterations=1500, learning_rate=0.03, depth=8, 
        cat_features=cat_features, verbose=0, random_seed=42, 
        eval_metric='R2', early_stopping_rounds=50
    )
    model_low.fit(X_train_l, y_train_l, eval_set=(X_test_l, y_test_l))
    
    # Değerlendirme Mod 1
    pred_low_log = model_low.predict(X_test_l)
    r2_low = r2_score(y_test_l, pred_low_log)
    print(f"✅ Model 1 R2 (Log): {r2_low:.4f}")
    
    # --- MODEL 2: HIGH SEVERITY (Ağır Ceza) ---
    print("\n🚀 MODEL 2: High Severity Eğitiliyor...")
    y_high = np.log1p(df_high['jail'])
    X_high = df_high[available_features]
    
    X_high_processed = X_high.copy()
    for col in cat_features:
        X_high_processed[col] = X_high_processed[col].fillna("Unknown").astype(str)

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_high_processed, y_high, test_size=0.2, random_state=42)
    
    # Ağır cezalar daha karmaşık olabilir, daha derin ağaç verelim
    model_high = CatBoostRegressor(
        iterations=1500, learning_rate=0.02, depth=10, 
        cat_features=cat_features, verbose=0, random_seed=42,
        eval_metric='R2', early_stopping_rounds=50, l2_leaf_reg=5 
    )
    model_high.fit(X_train_h, y_train_h, eval_set=(X_test_h, y_test_h))
    
    # Değerlendirme Mod 2
    pred_high_log = model_high.predict(X_test_h)
    r2_high = r2_score(y_test_h, pred_high_log)
    print(f"✅ Model 2 R2 (Log): {r2_high:.4f}")
    
    # --- GENEL PERFORMANS (AĞIRLIKLI ORTALAMA) ---
    print("\n🏆 GENEL PERFORMANS DEĞERLENDİRMESİ")
    
    # Test setlerini birleştirip bakmak en doğrusu ama burada basitçe ağırlıklı ortalamaya bakalım
    # Gerçek dünya simülasyonu için test setlerini birleştirip tek tek tahmin edip bakmak lazım
    
    # Simülasyon: Hangi modele gideceğini bildiğimizi varsayıyoruz (İdeal Senaryo)
    # (Web tarafında bir Sınıflandırıcı (Classifier) koyup önce "Ağır mı Hafif mi" diye soracağız)
    
    all_y_true = np.concatenate([np.expm1(y_test_l), np.expm1(y_test_h)])
    all_y_pred = np.concatenate([np.expm1(pred_low_log), np.expm1(pred_high_log)])
    all_y_true_log = np.concatenate([y_test_l, y_test_h])
    all_y_pred_log = np.concatenate([pred_low_log, pred_high_log])
    
    final_r2_log = r2_score(all_y_true_log, all_y_pred_log)
    final_r2_orig = r2_score(all_y_true, all_y_pred)
    final_mae = mean_absolute_error(all_y_true, all_y_pred)
    
    print(f"🔹 Genel R2 Score (Log Scale): {final_r2_log:.4f}")
    print(f"🔹 Genel R2 Score (Original): {final_r2_orig:.4f}")
    print(f"🔹 Genel MAE: {final_mae:.2f} gün")
    
    if final_r2_log > 0.77: # Önceki en iyi 0.76 idi
        print("\n🎉 MÜKEMMEL! Yeni bir rekor kırdık.")
        
    # Kayıt
    model_low.save_model(str(MODEL_DIR / "model_low_3000.cbm"))
    model_high.save_model(str(MODEL_DIR / "model_high_3000.cbm"))
    joblib.dump(available_features, MODEL_DIR / "features_list.pkl")
    joblib.dump(cat_features, MODEL_DIR / "cat_features_list.pkl")
    
    print(f"\n💾 Modeller kaydedildi: {MODEL_DIR}")

if __name__ == "__main__":
    train_segmented_models()
