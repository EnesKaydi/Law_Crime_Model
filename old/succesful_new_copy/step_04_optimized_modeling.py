
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("../outputs/model_results_optimized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("../model_data_optimized")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def optimized_training():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # 1. FİLTRELEME VE OUTLIER TEMİZLİĞİ
    if 'jail' not in df.columns:
        return
        
    print(f"📊 Orijinal Veri: {df.shape}")
    
    # Adım 1: 0-300 arasını çıkar
    df = df[df['jail'] > 300].copy()
    print(f"✅ Filtre (>300): {df.shape[0]} satır")
    
    # Adım 2: Extreme Outlier Temizliği (Mantıksız derecede yüksek cezalar)
    # %99.5 percentilesine bakalım
    ust_sinir = df['jail'].quantile(0.995)
    print(f"⚠️ %99.5 Üst Sınır: {ust_sinir} gün ({ust_sinir/365:.1f} yıl)")
    
    # 255500 gibi değerler hatalı veya müebbet olabilir, bunlar regresyonu bozar.
    # 50 yıl (18250 gün) üzeri çok nadirdir.
    # Veri setinde tutarlılık için bu aşırı uçları temizleyelim.
    df_clean = df[df['jail'] <= ust_sinir].copy()
    print(f"✅ Outlier Temizliği Sonrası: {df_clean.shape[0]} satır (Atılan: {len(df) - len(df_clean)})")
    
    # 2. FEATURE ENGINEERING (GENİŞLETİLMİŞ)
    print("\n⚙️ Feature Engineering (Genişletilmiş)...")
    
    y = np.log1p(df_clean['jail'])
    
    # Daha fazla feature ekle
    features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail'
    ]
    
    # prior_charges_severity kolonlarını bul ve ekle
    prior_severity_cols = [c for c in df.columns if 'prior_charges_severity' in c]
    features.extend(prior_severity_cols)
    print(f"➕ Eklenen 'prior_charges' sayısı: {len(prior_severity_cols)}")
    
    available_features = [f for f in features if f in df_clean.columns]
    X = df_clean[available_features].copy()
    
    # Missing Value Handling
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)
            
    # Encoding
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # 3. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. TARGET ENCODING (JUDGE ID)
    if 'judge_id' in X_train.columns:
        judge_means = y_train.groupby(X_train['judge_id']).mean()
        global_mean = y_train.mean()
        
        X_train['judge_mean_jail'] = X_train['judge_id'].map(judge_means).fillna(global_mean)
        X_test['judge_mean_jail'] = X_test['judge_id'].map(judge_means).fillna(global_mean)
        
        X_train.drop('judge_id', axis=1, inplace=True)
        X_test.drop('judge_id', axis=1, inplace=True)

    # 5. MODEL EĞİTİMİ (XGBoost - Optimize Edilmiş Parametreler)
    print("\n🚀 XGBoost Modeli Eğitiliyor (Optimize)...")
    
    # Daha derin ağaçlar ve daha yavaş öğrenme (daha iyi genelleme için)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,     # Arttırıldı
        learning_rate=0.02,    # Azaltıldı
        max_depth=8,           # Arttırıldı
        subsample=0.7,         # Overfit önlemi
        colsample_bytree=0.7,  # Feature seçimi
        reg_alpha=0.1,         # L1 Regularization
        reg_lambda=1.0,        # L2 Regularization
        n_jobs=-1,
        random_state=42
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # 6. DEĞERLENDİRME
    print("\n📊 Optimize Model Sonuçları:")
    
    y_pred_log = xgb_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    r2_log = r2_score(y_test, y_pred_log)
    r2_orig = r2_score(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    mae = mean_absolute_error(y_test_orig, y_pred)
    
    print(f"🔹 R2 Score (Log Scale): {r2_log:.4f}")
    print(f"🔹 R2 Score (Original Scale): {r2_orig:.4f}")
    print(f"🔹 RMSE: {rmse:.2f} gün")
    print(f"🔹 MAE: {mae:.2f} gün")
    
    if r2_log > 0.80:
        print("\n✅ HEDEF BAŞARILDI! (%80+)")
    else:
        print(f"\n⚠️ Hala hedefin altındayız (%{r2_log*100:.2f}).")
        
    # Feature Importance
    plt.figure(figsize=(12, 10))
    xgb.plot_importance(xgb_model, max_num_features=25, height=0.5)
    plt.title(f'Feature Importance (R2: {r2_log:.2f})')
    plt.savefig(OUTPUT_DIR / "optimized_importance.png")
    plt.close()

    # Modeli Kaydet
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_optimized.pkl")

if __name__ == "__main__":
    optimized_training()
