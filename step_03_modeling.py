
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
import joblib

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Yollar
VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/model_results_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("model_data_v1")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_models():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # 1. TEMİZLİK VE FİLTRELEME
    if 'jail' not in df.columns:
        print("❌ 'jail' kolonu yok!")
        return

    # 0-300 arasını çıkar (>300 olanlar kalsın)
    df = df[df['jail'] > 300].copy()
    print(f"✅ Veri Filtrelendi (jail > 300): {df.shape[0]} satır")
    
    # 2. FEATURE ENGINEERING (ÖNEMLİ KISIM)
    print("\n⚙️ Feature Engineering...")
    
    # Hedef Log Dönüşümü
    y = np.log1p(df['jail'])
    
    # Seçilen Önemli Özellikler (Analizden gelen)
    # highest_severity en önemli. Diğerleri: violent_crime, is_recid_new, wcisclass, release.
    # judge_id'yi target encoding için tutuyoruz ama feature olarak direkt değil.
    
    features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor'
    ]
    
    # Veri setinde var olan featureları seç
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    
    # Eksik verileri doldur
    # Kategorik ve Sayısal ayrımı
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)
            
    # Label Encoding (Kategorik için)
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # 3. SPLIT (Train/Test)
    # Target encoding sızıntısını önlemek için önce bölüyoruz!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"✅ Eğitim Seti: {X_train.shape}, Test Seti: {X_test.shape}")
    
    # 4. TARGET ENCODING (JUDGE ID)
    # Sadece Train seti üzerinde hesapla, Test setine uygula
    if 'judge_id' in X_train.columns:
        # Train üzerinde hesapla
        judge_means = y_train.groupby(X_train['judge_id']).mean()
        global_mean = y_train.mean()
        
        # Train'e ekle
        X_train['judge_mean_jail'] = X_train['judge_id'].map(judge_means)
        X_train['judge_mean_jail'].fillna(global_mean, inplace=True)
        
        # Test'e ekle (Train'den gelen map ile!)
        X_test['judge_mean_jail'] = X_test['judge_id'].map(judge_means)
        X_test['judge_mean_jail'].fillna(global_mean, inplace=True) # Testte yeni hakim varsa global mean
        
        # judge_id'yi çıkarabiliriz artık veya tutabiliriz (tree based modeller handle edebilir)
        # Şimdilik çıkarıp bias özelliğini kullanalım
        X_train.drop('judge_id', axis=1, inplace=True)
        X_test.drop('judge_id', axis=1, inplace=True)
        
        print("✅ Judge Target Encoding uygulandı (Leakage önlendi).")
        
    # 5. MODEL EĞİTİMİ (XGBoost)
    print("\n🚀 XGBoost Modeli Eğitiliyor...")
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    
    # 6. DEĞERLENDİRME
    print("\n📊 Model Performansı Değerlendiriliyor...")
    
    y_pred_log = xgb_model.predict(X_test)
    y_pred = np.expm1(y_pred_log) # Log'dan geri dön
    y_test_orig = np.expm1(y_test)
    
    # Metrikler
    r2 = r2_score(y_test, y_pred_log) # Log scale üzerinde R2 daha anlamlı olabilir lineer ilişki için
    r2_orig = r2_score(y_test_orig, y_pred) # Orijinal scale
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    mae = mean_absolute_error(y_test_orig, y_pred)
    
    print(f"🔹 R2 Score (Log Scale - Model Başarısı): {r2:.4f}")
    print(f"🔹 R2 Score (Original Scale): {r2_orig:.4f}")
    print(f"🔹 RMSE (Hata Kareleri Ort.): {rmse:.2f} gün")
    print(f"🔹 MAE (Ortalama Mutlak Hata): {mae:.2f} gün")
    
    # %80 Başarı Hedefi Kontrolü
    # R2 > 0.80 mi?
    if r2 > 0.80:
        print("\n✅ TEBRİKLER! %80 Başarı Hedefine Ulaşıldı (Log Scale R2).")
    else:
        print(f"\n⚠️ Hedefin altındayız (%{r2*100:.2f}). İyileştirme gerekebilir.")
        
    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(xgb_model, max_num_features=20, height=0.5)
    plt.title('XGBoost Feature Importance')
    plt.savefig(OUTPUT_DIR / "xgboost_importance.png")
    plt.close()
    
    # Tahmin vs Gerçek Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_orig, y_pred, alpha=0.3)
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.xlabel('Gerçek Ceza (Gün)')
    plt.ylabel('Tahmin Edilen Ceza (Gün)')
    plt.title(f'Tahmin vs Gerçek (R2: {r2:.2f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(OUTPUT_DIR / "prediction_scatter.png")
    plt.close()
    
    # Modeli Kaydet
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_model_v1.pkl")
    print(f"\n💾 Model ve sonuçlar kaydedildi.")

if __name__ == "__main__":
    # Gerekli kütüphane kontrolü (xgboost)
    try:
        import xgboost
    except ImportError:
        print("XGBoost eksik, kuruluyor...")
        import os
        os.system("pip install xgboost")
        
    train_models()
