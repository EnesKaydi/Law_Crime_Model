
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
import joblib
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("model_data_v2_interactions")
OUTPUT_DIR = Path("outputs/geo_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_geo_justice():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Filtreleme (Mainstream + High dahil) - 300 gün üstü
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    print(f"✅ Analiz Verisi: {len(df)} satır")
    
    # --- 1. HAM İSTATİSTİKLER (Suç Haritası) ---
    print("\n🗺️ İlçe (County) Bazlı Suç İstatistikleri Hesaplanıyor...")
    
    county_stats = df.groupby('county').agg({
        'jail': 'mean',
        'highest_severity': 'mean',
        'violent_crime': 'mean',
        'age_offense': 'mean',
        'case_type': 'count'
    }).rename(columns={'case_type': 'count'})
    
    # Az verisi olan ilçeleri ele (En az 50 dava)
    county_stats = county_stats[county_stats['count'] > 50].sort_values(by='jail', ascending=False)
    
    print("🏆 En Ağır Cezaların Verildiği 10 İlçe:")
    print(county_stats[['jail', 'highest_severity', 'count']].head(10))
    
    # --- 2. ADALET ANALİZİ (Model Residuals - Hakkaniyet) ---
    # Modelin tahmin ettiğinden FAZLA ceza veren ilçeler "Sert", AZ verenler "Yumuşak".
    # Bu analiz için V2 Mainstream modelini kullanalım (Çoğunluk verisi).
    
    if not MODEL_DIR.exists():
        print("⚠️ Model klasörü yok, sadece ham istatistiklerle devam ediliyor.")
        return

    # Özellik Hazırlığı
    features = joblib.load(MODEL_DIR / "features_v2.pkl")
    cat_features = joblib.load(MODEL_DIR / "cat_features_v2.pkl")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_DIR / "model_low_v2.cbm")) # Mainstream model
    
    # Model girdilerini hazırla
    # Sadece Mainstream verisi (3000 gün altı) üzerinde residual bakmak daha sağlıklı
    df_main = df[df['jail'] <= 3000].copy()
    
    if 'highest_severity' in df_main.columns and 'violent_crime' in df_main.columns:
        df_main['severity_x_violent'] = df_main['highest_severity'] * df_main['violent_crime']
    if 'age_judge' in df_main.columns and 'age_offense' in df_main.columns:
        df_main['age_judge'] = df_main['age_judge'].fillna(df_main['age_judge'].mean())
        df_main['age_offense'] = df_main['age_offense'].fillna(df_main['age_offense'].mean())
        df_main['age_gap'] = df_main['age_judge'] - df_main['age_offense']
    if 'is_recid_new' in df_main.columns and 'violent_crime' in df_main.columns:
        df_main['violent_recid'] = df_main['is_recid_new'] * df_main['violent_crime']

    X = df_main[features].copy()
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
            
    print("⏳ Model Tahminleri Alınıyor (Sertlik Skoru İçin)...")
    preds_log = model.predict(X)
    df_main['predicted_jail'] = np.expm1(preds_log)
    
    # Residual (Gerçek - Tahmin)
    # Pozitif Residual: Beklenenden çok ceza (Sert)
    # Negatif Residual: Beklenenden az ceza (Yumuşak)
    df_main['residual'] = df_main['jail'] - df_main['predicted_jail']
    
    geo_justice = df_main.groupby('county')['residual'].mean().sort_values(ascending=False)
    geo_justice_count = df_main['county'].value_counts()
    geo_justice = geo_justice[geo_justice_count > 50] # Filtre
    
    print("\n⚖️ En 'Acımasız' 5 İlçe (Tahmin edilenden fazla ceza):")
    print(geo_justice.head(5))
    
    print("\n🕊️ En 'Hoşgörülü' 5 İlçe (Tahmin edilenden az ceza):")
    print(geo_justice.tail(5))
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    geo_justice.head(10).plot(kind='barh', color='darkred', label='Sert (Fazla Ceza)')
    geo_justice.tail(10).plot(kind='barh', color='darkgreen', label='Yumuşak (Az Ceza)') # Üst üste binmemesi için ayrı plot lazım ama basitlik için:
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    top_bottom = pd.concat([geo_justice.head(10), geo_justice.tail(10)])
    colors = ['red' if x > 0 else 'green' for x in top_bottom.values]
    top_bottom.plot(kind='barh', color=colors, ax=ax)
    plt.title('Coğrafi Adalet: İlçelerin Sertlik Skorları (Residuals)')
    plt.xlabel('Ortalama Sapma (Gün)')
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "geo_justice_score.png")
    
    print(f"\n💾 Harita grafiği kaydedildi: {OUTPUT_DIR}")
    
    # CSV Kaydet
    geo_justice.to_csv(OUTPUT_DIR / "county_harshness_score.csv")

if __name__ == "__main__":
    analyze_geo_justice()
