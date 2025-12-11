
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import joblib
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/retrain_interactions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("model_data_v2_interactions")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def create_interactions(df):
    """Yeni keşfedilen etkileşim özelliklerini türetir."""
    # 1. Severity x Violent (Süper Kombinasyon)
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
        
    # 2. Age Gap (Kuşak Çatışması)
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        # Eksikleri doldur
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
        
    # 3. Violent Recidivist (Şiddetli Tekerrür)
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
        
    return df

def retrain_all_models():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Temel Filtreleme
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # YENİ ÖZELLİKLERİ EKLE
    print("✨ Yeni 'Interaction Features' ekleniyor...")
    df = create_interactions(df)
    
    # Özellik Listesi
    base_features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    base_features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    
    # Yeni eklenenler
    new_features = ['severity_x_violent', 'age_gap', 'violent_recid']
    all_features = base_features + new_features
    
    # Kullanılabilir olanları seç
    final_features = [f for f in all_features if f in df.columns]
    
    # Kategorik Belirleme
    cat_features = []
    X = df[final_features].copy()
    
    # Kategorikleri işle (Zorla String)
    KNOWN_CAT_FEATURES = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass']
    
    for col in X.columns:
        if col in KNOWN_CAT_FEATURES or X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("Unknown").astype(str)
            if col not in cat_features: cat_features.append(col)
            
    # Sayısallar için basit fillna
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
            
    print(f"📌 Son Özellik Sayısı: {len(final_features)}")
    print(f"📌 Yeni Eklenenler: {new_features}")
    
    # --- 1. ROUTER EĞİTİMİ (V2) ---
    print("\n🚀 Router Model V2 Eğitiliyor...")
    THRESHOLD = 3000
    y_router = (df['jail'] > THRESHOLD).astype(int)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_router, test_size=0.2, random_state=42, stratify=y_router)
    
    router = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, cat_features=cat_features,
        verbose=0, random_seed=42, auto_class_weights='Balanced', eval_metric='F1'
    )
    router.fit(X_train_r, y_train_r)
    
    r_pred = router.predict(X_test_r)
    r_acc = accuracy_score(y_test_r, r_pred)
    r_f1 = f1_score(y_test_r, r_pred)
    print(f"✅ Router V2 Accuracy: %{r_acc*100:.2f} (Önceki: %87.89)")
    print(f"✅ Router V2 F1 Score: %{r_f1*100:.2f} (Önceki: %52.55)")
    
    # --- 2. SEGMENT MODELLERİ EĞİTİMİ (V2) ---
    mask_low = df['jail'] <= THRESHOLD
    mask_high = df['jail'] > THRESHOLD
    
    # Model Low V2
    print("\n🚀 Model Low V2 (Mainstream) Eğitiliyor...")
    X_low = X[mask_low]
    y_low = np.log1p(df[mask_low]['jail'])
    
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_low, y_low, test_size=0.2, random_state=42)
    
    model_low = CatBoostRegressor(
        iterations=1500, learning_rate=0.03, depth=8, cat_features=cat_features,
        verbose=0, random_seed=42, eval_metric='R2', early_stopping_rounds=50
    )
    model_low.fit(X_train_l, y_train_l)
    r2_low = r2_score(y_test_l, model_low.predict(X_test_l))
    print(f"✅ Model Low V2 R2: {r2_low:.4f} (Önceki: 0.7033)")
    
    # Model High V2
    print("\n🚀 Model High V2 (Severity) Eğitiliyor...")
    X_high = X[mask_high]
    y_high = np.log1p(df[mask_high]['jail'])
    
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_high, y_high, test_size=0.2, random_state=42)
    
    model_high = CatBoostRegressor(
        iterations=1500, learning_rate=0.02, depth=10, cat_features=cat_features,
        verbose=0, random_seed=42, eval_metric='R2', early_stopping_rounds=50, l2_leaf_reg=5
    )
    model_high.fit(X_train_h, y_train_h)
    r2_high = r2_score(y_test_h, model_high.predict(X_test_h))
    print(f"✅ Model High V2 R2: {r2_high:.4f} (Önceki: 0.3274)")
    
    # --- GENEL PERFORMANS (SİMÜLASYON) ---
    print("\n🏆 GENEL PERFORMANS V2")
    # Basit ağırlıklı ortalama ile genel skoru tahmin edelim
    all_y_true_log = np.concatenate([y_test_l, y_test_h])
    all_y_pred_log = np.concatenate([model_low.predict(X_test_l), model_high.predict(X_test_h)])
    all_y_true = np.expm1(all_y_true_log)
    all_y_pred = np.expm1(all_y_pred_log)
    
    final_r2_log = r2_score(all_y_true_log, all_y_pred_log)
    final_r2_orig = r2_score(all_y_true, all_y_pred)
    final_mae = mean_absolute_error(all_y_true, all_y_pred)
    
    print(f"🔹 Genel R2 Score (Log Scale): {final_r2_log:.4f} (Hedef: >0.83)")
    print(f"🔹 Genel R2 Score (Original): {final_r2_orig:.4f}")
    print(f"🔹 Genel MAE: {final_mae:.2f} gün")
    
    # Kayıt
    router.save_model(str(MODEL_DIR / "router_v2.cbm"))
    model_low.save_model(str(MODEL_DIR / "model_low_v2.cbm"))
    model_high.save_model(str(MODEL_DIR / "model_high_v2.cbm"))
    joblib.dump(final_features, MODEL_DIR / "features_v2.pkl")
    joblib.dump(cat_features, MODEL_DIR / "cat_features_v2.pkl")
    
    print(f"\n💾 V2 Modelleri Kaydedildi: {MODEL_DIR}")

if __name__ == "__main__":
    retrain_all_models()
