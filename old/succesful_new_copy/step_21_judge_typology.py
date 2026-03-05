
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("../model_data_v2_interactions")
OUTPUT_DIR = Path("../outputs/judge_typology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_judges():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı
    if 'jail' not in df.columns: return
    df = df[df['jail'].between(300, 3000)].copy() # Mainstream odaklı
    
    # Model Yükle (Residual Hesabı İçin)
    if not MODEL_DIR.exists():
        print("❌ Model klasörü yok!")
        return
        
    features = joblib.load(MODEL_DIR / "features_v2.pkl")
    cat_features = joblib.load(MODEL_DIR / "cat_features_v2.pkl")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_DIR / "model_low_v2.cbm"))
    
    # Feature Engineering (Model için)
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        df['age_judge'] = df['age_judge'].fillna(df['age_judge'].mean())
        df['age_offense'] = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = df['age_judge'] - df['age_offense']
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
        
    # Tahmin Al
    X = df[features].copy()
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
            
    print("⏳ Model Tahminleri Alınıyor (Hakim Skoru İçin)...")
    preds_log = model.predict(X)
    df['predicted_jail'] = np.expm1(preds_log)
    df['residual'] = df['jail'] - df['predicted_jail'] # Pozitif = Sert
    
    # --- YARGIÇ PROFİLİ OLUŞTURMA ---
    print("\n⚖️ Hakim Profilleri Çıkarılıyor...")
    
    judge_stats = df.groupby('judge_id').agg({
        'jail': 'mean',                  # Ortalama Ceza
        'residual': 'mean',              # Sertlik Skoru (Bias)
        'highest_severity': 'mean',      # Baktığı Davaların Ağırlığı
        'violent_crime': 'mean',         # Şiddet Davası Oranı (%)
        'case_type': 'count'             # Dava Sayısı
    })
    
    # Az davası olan hakimleri ele (En az 50 dava)
    judge_stats = judge_stats[judge_stats['case_type'] > 50].copy()
    
    # Kümeleme (K-Means)
    print("🔄 Hakimler Kümeleniyor (Judge Clustering)...")
    
    # Clustering Features
    X_judge = judge_stats[['residual', 'highest_severity', 'violent_crime']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_judge)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    judge_stats['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Profilleri İsimlendir
    profile = judge_stats.groupby('cluster').mean()
    print("\n👥 Hakim Tipi Profilleri (Ortalamalar):")
    print(profile)
    
    # Görselleştirme
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='residual', 
        y='highest_severity', 
        hue='cluster', 
        size='violent_crime',
        data=judge_stats, 
        palette='deep',
        alpha=0.7
    )
    plt.title('Yargıç Tipolojisi: Sertlik vs Dava Ağırlığı')
    plt.xlabel('Sertlik Skoru (Fazladan Verilen Gün)')
    plt.ylabel('Ortalama Dava Ağırlığı (Severity)')
    plt.axvline(0, color='gray', linestyle='--')
    plt.savefig(OUTPUT_DIR / "judge_clusters.png")
    
    # En Sert ve En Yumuşak Hakimler
    top_harsh = judge_stats.sort_values(by='residual', ascending=False).head(5)
    top_lenient = judge_stats.sort_values(by='residual', ascending=True).head(5)
    
    print("\n🔥 En Acımasız 5 Hakim (The Hammer):")
    print(top_harsh[['residual', 'jail', 'case_type']])
    
    print("\n🕊️ En Babacan 5 Hakim (The Dove):")
    print(top_lenient[['residual', 'jail', 'case_type']])
    
    # Kaydet
    judge_stats.to_csv(OUTPUT_DIR / "judge_profiles.csv")
    print(f"\n💾 Yargıç profilleri kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_judges()
