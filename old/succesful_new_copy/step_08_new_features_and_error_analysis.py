
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostRegressor
import joblib
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_PATH = Path("../model_data_advanced/catboost_model.cbm")
FEATURES_PATH = Path("../model_data_advanced/features_list.pkl")
OUTPUT_DIR = Path("../outputs/optimization_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_new_features_and_errors():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı (Aynı filtreler)
    if 'jail' not in df.columns:
        return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # 1. YENİ ÖZELLİKLERİN KEŞFİ
    print("\n🔍 Yeni Özellik Analizi (County, Case Type, Zip)")
    new_features = ['county', 'case_type', 'zip']
    
    for col in new_features:
        if col in df.columns:
            print(f"\nKolon: {col}")
            print(f"  • Unique Değer Sayısı: {df[col].nunique()}")
            print(f"  • En Sık Görülen 5 Değer:")
            print(df[col].value_counts().head(5))
            
            # Hedef değişkenle ilişkisi (ANOVA veya Görsel)
            # Çok fazla kategori varsa sadece en sık görülenlere bak
            top_categories = df[col].value_counts().head(20).index
            subset = df[df[col].isin(top_categories)]
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col, y='jail', data=subset)
            plt.xticks(rotation=45)
            plt.title(f'{col} vs Ceza (En Sık 20)')
            plt.savefig(OUTPUT_DIR / f"{col}_jail_relation.png")
            plt.close()
            print(f"  ✅ Grafik kaydedildi: {OUTPUT_DIR}/{col}_jail_relation.png")
            
            # Eksik değer kontrolü
            null_count = df[col].isnull().sum()
            print(f"  • Eksik Değer: {null_count} (%{null_count/len(df)*100:.2f})")
        else:
            print(f"❌ Kolon bulunamadı: {col}")

    # 2. HATA ANALİZİ (Mevcut Model Nerede Yanılıyor?)
    print("\n🕵️ Hata Analizi (Error Analysis)")
    
    if not MODEL_PATH.exists():
        print("⚠️ Önceki model bulunamadı, hata analizi atlanıyor.")
        return
        
    # Modeli Yükle
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    feature_names = joblib.load(FEATURES_PATH)
    
    # Tahmin için X hazırla
    X = df[feature_names].copy()
    
    # Kategorik dönüşüm (Inference'daki gibi)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna("Unknown").astype(str)
            X.loc[X[col] == 'nan', col] = "Unknown"
            
    # Tahmin Yap
    print("⏳ Tahmin yapılıyor...")
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    
    # Hataları Hesapla
    df['prediction'] = y_pred
    df['error'] = df['prediction'] - df['jail'] # Pozitif: Fazla tahmin, Negatif: Az tahmin
    df['abs_error'] = df['error'].abs()
    
    # En Kötü 20 Tahmin
    print("\n📉 En Kötü 10 Tahmin (En Yüksek Hata):")
    worst_predictions = df.sort_values('abs_error', ascending=False).head(10)
    print(worst_predictions[['jail', 'prediction', 'error', 'highest_severity', 'judge_id', 'county', 'case_type']])
    
    # Hata hangi özellikte yoğunlaşıyor? (Örn: Belirli bir case_type'da mı?)
    if 'case_type' in df.columns:
        error_by_case = df.groupby('case_type')['abs_error'].mean().sort_values(ascending=False)
        print("\n⚠️ 'case_type' Bazlı Ortalama Hata (İlk 10):")
        print(error_by_case.head(10))
        
    if 'county' in df.columns:
        error_by_county = df.groupby('county')['abs_error'].mean().sort_values(ascending=False)
        print("\n⚠️ 'county' Bazlı Ortalama Hata (İlk 10):")
        print(error_by_county.head(10))

    # Hatanın Dağılımı
    plt.figure(figsize=(10, 6))
    sns.histplot(df['error'], bins=100)
    plt.title('Hata Dağılımı (Tahmin - Gerçek)')
    plt.xlabel('Hata (Gün)')
    plt.xlim(-2000, 2000) # Aşırı uçları görmezden gel
    plt.savefig(OUTPUT_DIR / "error_distribution.png")
    plt.close()
    
    print(f"\n💾 Analiz sonuçları {OUTPUT_DIR} konumuna kaydedildi.")

if __name__ == "__main__":
    analyze_new_features_and_errors()
