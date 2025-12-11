
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/v3_persona_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("model_data_v3_persona")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def create_interactions(df):
    """V2'den gelen etkileşim özellikleri."""
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
        
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
        
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
    return df

def train_persona_model():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # 1. Filtreleme ve Hazırlık
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # Etkileşimler
    df = create_interactions(df)
    
    # 2. Pipeline Split (Train/Test)
    # Veri sızıntısını (Leakage) önlemek için Cluster modelini sadece X_train üzerinde eğiteceğiz!
    X_full = df.drop(columns=['jail'])
    y_full = df['jail']
    
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    # 3. PERSONA MODELİ (K-MEANS ENTEGRASYONU)
    print("\n🕵️ Persona Dedektörü (K-Means) Eğitiliyor...")
    
    # Clustering için kullanılacak özellikler (Ham veri)
    cluster_cols = ['age_offense', 'highest_severity', 'prior_felony', 'prior_misdemeanor', 'violent_crime']
    
    # Eksikleri doldur (Clustering için)
    X_train_cl = X_train[cluster_cols].fillna(X_train[cluster_cols].mean())
    X_test_cl = X_test[cluster_cols].fillna(X_train[cluster_cols].mean()) # Train ortalamasıyla doldur!
    
    # Scale Et
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cl)
    X_test_scaled = scaler.transform(X_test_cl) # Train ile scale et!
    
    # Eğit
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)
    
    # Modele "feature" olarak ekle
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['persona_cluster'] = train_clusters.astype(str) # Kategorik yap
    X_test['persona_cluster'] = test_clusters.astype(str)
    
    print(f"✅ Persona Tiplemesi Tamamlandı. (Örnek: {X_train['persona_cluster'].iloc[0]})")
    
    # 4. FINAL FEATURES
    base_features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip',
        'severity_x_violent', 'age_gap', 'violent_recid',
        'persona_cluster' # YENİ YILDIZIMIZ
    ]
    base_features.extend([c for c in X_train.columns if 'prior_charges_severity' in c])
    final_features = [f for f in base_features if f in X_train.columns]
    
    cat_features = []
    
    # Kategorik İşlemleri (Train/Test için ortak)
    KNOWN_CAT_FEATURES = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass', 'persona_cluster']
    
    for col in final_features:
        if col in KNOWN_CAT_FEATURES or X_train[col].dtype == 'object':
            X_train[col] = X_train[col].fillna("Unknown").astype(str)
            X_test[col] = X_test[col].fillna("Unknown").astype(str)
            if col not in cat_features: cat_features.append(col)
        else:
            mean_val = X_train[col].mean()
            X_train[col] = X_train[col].fillna(mean_val)
            X_test[col] = X_test[col].fillna(mean_val) # Test de mean ile dolsun
            
    # 5. MODEL EĞİ TİMLERİ (V3)
    print(f"\n🚀 Modeller Eğitiliyor (Özellik Sayısı: {len(final_features)})...")
    
    # --- ROUTER V3 ---
    THRESHOLD = 3000
    y_train_r = (y_train > THRESHOLD).astype(int)
    y_test_r = (y_test > THRESHOLD).astype(int)
    
    router = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, cat_features=cat_features,
        verbose=0, random_seed=42, auto_class_weights='Balanced', eval_metric='F1'
    )
    router.fit(X_train[final_features], y_train_r)
    
    r_pred = router.predict(X_test[final_features])
    print(f"✅ Router V3 Accuracy: %{accuracy_score(y_test_r, r_pred)*100:.2f}")
    
    # --- SEGMENT MODELLERİ (Log Scale) ---
    # Train setini böl
    mask_low_train = y_train <= THRESHOLD
    mask_high_train = y_train > THRESHOLD
    
    # Test setini böl (Değerlendirme için)
    mask_low_test = y_test <= THRESHOLD
    mask_high_test = y_test > THRESHOLD
    
    # Model Low V3
    print("🚀 Model Low V3 Eğitiliyor...")
    model_low = CatBoostRegressor(
        iterations=1500, learning_rate=0.03, depth=8, cat_features=cat_features,
        verbose=0, random_seed=42, eval_metric='R2', early_stopping_rounds=50
    )
    model_low.fit(X_train[mask_low_train][final_features], np.log1p(y_train[mask_low_train]))
    
    # Model High V3
    print("🚀 Model High V3 Eğitiliyor...")
    model_high = CatBoostRegressor(
        iterations=1500, learning_rate=0.02, depth=10, cat_features=cat_features,
        verbose=0, random_seed=42, eval_metric='R2', early_stopping_rounds=50, l2_leaf_reg=5
    )
    model_high.fit(X_train[mask_high_train][final_features], np.log1p(y_train[mask_high_train]))
    
    # 6. GENEL PERFORMANS TESTİ (PIPELINE SİMÜLASYONU)
    print("\n🏆 V3 GENEL PERFORMANS (Gerçek Test Seti Üzerinde)")
    
    # Test setindeki her veri için: Router -> Model Seç -> Tahmin
    final_preds_log = []
    
    router_preds = router.predict(X_test[final_features]) # 0 veya 1
    
    # Vektörize işlem yerine döngüyle simüle edelim (daha anlaşılır)
    # Ama hız için numpy indexing kullanalım
    
    # Düşük Tahminleri Al
    pred_low = model_low.predict(X_test[final_features])
    # Yüksek Tahminleri Al
    pred_high = model_high.predict(X_test[final_features])
    
    # Router'a göre seç
    final_preds_log = np.where(router_preds == 1, pred_high, pred_low)
    
    # Skorlar
    y_test_log = np.log1p(y_test)
    final_r2_log = r2_score(y_test_log, final_preds_log)
    final_r2_orig = r2_score(y_test, np.expm1(final_preds_log))
    final_mae = mean_absolute_error(y_test, np.expm1(final_preds_log))
    
    print(f"🔹 Genel R2 Score (Log Scale): {final_r2_log:.4f} (V2: 0.8306)")
    print(f"🔹 Genel R2 Score (Original): {final_r2_orig:.4f} (V2: 0.7907)")
    print(f"🔹 Genel MAE: {final_mae:.2f} gün (V2: 348)")
    
    # 7. KAYIT
    joblib.dump(kmeans, MODEL_DIR / "kmeans_v3.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_v3.pkl") # Scaler'ı unutma!
    router.save_model(str(MODEL_DIR / "router_v3.cbm"))
    model_low.save_model(str(MODEL_DIR / "model_low_v3.cbm"))
    model_high.save_model(str(MODEL_DIR / "model_high_v3.cbm"))
    joblib.dump(final_features, MODEL_DIR / "features_v3.pkl")
    joblib.dump(cat_features, MODEL_DIR / "cat_features_v3.pkl")
    
    print(f"\n💾 V3 Sistemi (Persona-Enabled) Kaydedildi: {MODEL_DIR}")

if __name__ == "__main__":
    train_persona_model()
