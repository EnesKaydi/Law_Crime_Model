
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from pathlib import Path
import joblib
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("../model_data_v2_interactions")
OUTPUT_DIR = Path("../outputs/bias_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_models_and_features():
    if not MODEL_DIR.exists():
        print("❌ Model klasörü bulunamadı! Önce modelleri eğitin.")
        return None, None, None, None, None
        
    router = joblib.load(MODEL_DIR / "cat_features_v2.pkl") # Placeholder check
    features = joblib.load(MODEL_DIR / "features_v2.pkl")
    cat_features = joblib.load(MODEL_DIR / "cat_features_v2.pkl")
    
    # Modelleri yüklemek yerine tahminleri sıfırdan yapmak daha temiz olabilir
    # Ama burada script içinde tekrar predict logic kurmak uzun.
    # Kolaylık olsun diye: Modeli yükleyip tahmin alacağız.
    # Ancak pipeline karmaşık (Router + 2 Model).
    # Basitlik adına: 'step_16' scripti zaten tahminleri CSV'ye dökseydi iyiydi.
    # Neyse, burada basitçe Mainstream (Model Low) üzerinden bias bakalım.
    # Çoğunluk veri (%92) orada olduğu için bias asıl orada aranmalı.
    
    model_low = CatBoostRegressor()
    model_low.load_model(str(MODEL_DIR / "model_low_v2.cbm"))
    
    return model_low, features, cat_features

def analyze_bias():
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
    
    # 3000 Altı Veri (Mainstream Bias Analizi)
    df_main = df[df['jail'] <= 3000].copy()
    print(f"✅ Analiz Kapsamı: 300-3000 gün arası {len(df_main)} dava (Verinin %92'si)")
    
    # Interaction Features Ekle (Model bunları bekliyor)
    if 'highest_severity' in df_main.columns and 'violent_crime' in df_main.columns:
        df_main['severity_x_violent'] = df_main['highest_severity'] * df_main['violent_crime']
    if 'age_judge' in df_main.columns and 'age_offense' in df_main.columns:
        df_main['age_judge'] = df_main['age_judge'].fillna(df_main['age_judge'].mean())
        df_main['age_offense'] = df_main['age_offense'].fillna(df_main['age_offense'].mean())
        df_main['age_gap'] = df_main['age_judge'] - df_main['age_offense']
    if 'is_recid_new' in df_main.columns and 'violent_crime' in df_main.columns:
        df_main['violent_recid'] = df_main['is_recid_new'] * df_main['violent_crime']
        
    model, features, cat_features = load_models_and_features()
    if model is None: return

    # Kategorik Hazırlık
    X = df_main[features].copy()
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
            
    # Sayısal Hazırlık
    for col in X.columns:
        if col not in cat_features:
             X[col] = X[col].fillna(X[col].mean())

    print("⏳ Tahminler alınıyor...")
    preds_log = model.predict(X)
    df_main['predicted_jail'] = np.expm1(preds_log)
    df_main['error'] = df_main['predicted_jail'] - df_main['jail']
    df_main['abs_error'] = df_main['error'].abs()
    
    # --- 1. IRK ANALİZİ (RACE BIAS) ---
    print("\n🌍 IRK ANALİZİ (Race Bias):")
    race_stats = df_main.groupby('race').agg({
        'jail': 'mean',
        'predicted_jail': 'mean',
        'abs_error': 'mean',
        'highest_severity': 'mean' # Suç ağırlığı kontrolü için
    }).sort_values(by='jail', ascending=False)
    
    race_stats['count'] = df_main['race'].value_counts()
    race_stats = race_stats[race_stats['count'] > 100] # Azınlıkları filtrele
    
    print(f"{'Irk':<15} | {'Adet':<6} | {'Gerçek Ort.':<12} | {'Tahmin Ort.':<12} | {'Fark (Bias)':<12} | {'Hata (MAE)':<10}")
    print("-" * 80)
    for index, row in race_stats.iterrows():
        bias = row['predicted_jail'] - row['jail']
        print(f"{index:<15} | {int(row['count']):<6} | {row['jail']:<12.1f} | {row['predicted_jail']:<12.1f} | {bias:<12.1f} | {row['abs_error']:<10.1f}")

    # Grafik: Race Bias
    plt.figure(figsize=(10, 6))
    sns.barplot(x=race_stats.index, y=race_stats['predicted_jail'] - race_stats['jail'])
    plt.title('Irklara Göre Model Önyargısı (Tahmin - Gerçek)')
    plt.ylabel('Gün Farkı (+ Fazla Ceza, - Az Ceza)')
    plt.axhline(0, color='black', linestyle='--')
    plt.savefig(OUTPUT_DIR / "race_bias.png")
    
    # --- 2. CİNSİYET ANALİZİ (SEX BIAS) ---
    print("\n👫 CİNSİYET ANALİZİ (Sex Bias):")
    sex_stats = df_main.groupby('sex').agg({
        'jail': 'mean',
        'predicted_jail': 'mean',
        'abs_error': 'mean'
    })
    print(sex_stats)
    
    # --- 3. KOŞULLU BIAS (Conditional Bias - Severity Kontrollü) ---
    # "Siyahiler daha çok ceza alıyor çünkü daha ağır suç işliyorlar" tezini test edelim.
    # Şiddet Skoru (Severity) eşitlendiğinde durum ne?
    
    print("\n⚖️ KOŞULLU BIAS (Aynı Suç Şiddetinde Irk Ayrımı Var mı?):")
    plt.figure(figsize=(12, 6))
    
    # İsimleri Düzelt (Veride 'African American' ve 'Caucasian' geçiyor)
    race_map = {'African American': 'Black', 'Caucasian': 'White'}
    df_main['race_mapped'] = df_main['race'].map(race_map)
    
    # Sadece Black ve White alalım
    df_bw = df_main[df_main['race_mapped'].isin(['Black', 'White'])]
    
    sns.lineplot(x='highest_severity', y='predicted_jail', hue='race_mapped', data=df_bw, marker='o')
    plt.title('Suç Şiddetine Göre Ceza Tahmini: Siyahi vs Beyaz')
    plt.xlabel('Suç Şiddeti (Severity)')
    plt.ylabel('Tahmin Edilen Ceza (Gün)')
    plt.savefig(OUTPUT_DIR / "conditional_bias_race.png")
    
    # Sayısal Karşılaştırma (Severity Gruplarında Fark)
    print("\n   [Severity Bazlı Siyahi-Beyaz Farkı]")
    df_bw['sev_bin'] = pd.cut(df_bw['highest_severity'], bins=[0, 3, 6, 9, 20], labels=['Düşük', 'Orta', 'Yüksek', 'Çok Yeksek'])
    
    pivot = df_bw.pivot_table(index='sev_bin', columns='race_mapped', values='predicted_jail', aggfunc='mean')
    pivot['Fark (Black - White)'] = pivot['Black'] - pivot['White']
    print(pivot)
    
    print(f"\n💾 Tüm analiz grafikleri kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_bias()
