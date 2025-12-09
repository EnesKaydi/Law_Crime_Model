
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Yollar
VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/new_analysis_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def advanced_processing():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # 1. FİLTRELEME (0-300 arasını at)
    # Kullanıcı 0-300 arasını "örneklemden çıkaracağız" dedi. jail > 300 olanları alıyoruz.
    if 'jail' in df.columns:
        df_filtered = df[df['jail'] > 300].copy()
        print(f"✅ Filtrelenmiş Veri (jail > 300): {df_filtered.shape[0]} satır")
    else:
        print("❌ 'jail' kolonu yok!")
        return

    # 2. TUTARSIZLIK ANALİZİ (Yıl ve Hakim)
    print("\n🔍 Tutarsızlık Analizi Başlıyor...")
    
    # Yıl Analizi
    if 'year' in df_filtered.columns:
        year_stats = df_filtered.groupby('year')['jail'].agg(['mean', 'median', 'count']).sort_index()
        print("\n📅 Yıllara Göre Ceza İstatistikleri:")
        print(year_stats)
        
        # Yıl bazlı trendi görselleştir
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_filtered, x='year', y='jail', estimator='median', errorbar=None, label='Medyan Ceza')
        sns.lineplot(data=df_filtered, x='year', y='jail', estimator='mean', errorbar=None, label='Ortalama Ceza')
        plt.title('Yıllara Göre Ceza Değişimi')
        plt.xlabel('Yıl')
        plt.ylabel('Ceza (Gün)')
        plt.legend()
        plt.savefig(OUTPUT_DIR / "year_trend.png")
        plt.close()
    
    # Hakim Analizi
    if 'judge_id' in df_filtered.columns:
        judge_stats = df_filtered.groupby('judge_id')['jail'].agg(['mean', 'median', 'count', 'std'])
        judge_stats['strictness'] = (judge_stats['mean'] - df_filtered['jail'].mean()) / df_filtered['jail'].std()
        
        print("\n⚖️ Hakim İstatistikleri (İlk 10):")
        print(judge_stats.sort_values('count', ascending=False).head(10))
        
        # Hakim tutarsızlığını görselleştir (En az 50 davası olanlar)
        active_judges = judge_stats[judge_stats['count'] > 50]
        plt.figure(figsize=(12, 6))
        sns.histplot(active_judges['mean'], kde=True)
        plt.title('Hakimlerin Ortalama Ceza Dağılımı (Strictness)')
        plt.xlabel('Ortalama Ceza (Gün)')
        plt.savefig(OUTPUT_DIR / "judge_distribution.png")
        plt.close()
        
        # Feature Engineering: Hakim sertlik skoru ekle
        # Target encoding mantığı (Bunu yaparken data leakage riskine dikkat etmeliyiz ama analiz için şimdilik tüm veri üzerinde)
        judge_map = judge_stats['mean'].to_dict()
        df_filtered['judge_mean_jail'] = df_filtered['judge_id'].map(judge_map)
        print("✅ 'judge_mean_jail' özelliği eklendi (Hakim Tutumu).")

    # 3. VERİ ÖN İŞLEME VE NORMALİZASYON
    print("\n⚙️ Veri Ön İşleme ve Normalizasyon...")
    
    # Hedef Değişken Dönüşümü: Log Transform (Çünkü çok çarpık)
    df_filtered['jail_log'] = np.log1p(df_filtered['jail'])
    print("✅ Hedef değişken Log dönüşümü yapıldı ('jail_log').")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df_filtered['jail'], bins=50)
    plt.title('Orijinal Jail Dağılımı')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df_filtered['jail_log'], bins=50)
    plt.title('Log Transformed Jail Dağılımı')
    plt.savefig(OUTPUT_DIR / "target_transform.png")
    plt.close()

    # Kategorik Değişkenleri Belirle
    cat_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    
    # Eksik verileri doldur (Basit bir strateji: sayısal -> medyan, kategorik -> mode)
    # Ancak önce çok eksik olanları çıkaralım mı?
    # %50'den fazla eksik olanları raporlamıştık. 'recid_180d' vs.
    # Şimdilik basit doldurma yapalım, korelasyonu görmek için.
    
    for col in num_cols:
        if df_filtered[col].isnull().sum() > 0:
            df_filtered[col].fillna(df_filtered[col].median(), inplace=True)
            
    for col in cat_cols:
        if df_filtered[col].isnull().sum() > 0:
            df_filtered[col].fillna(df_filtered[col].mode()[0], inplace=True)
            
    # Encoding (Label Encoding for correlation analysis)
    le = LabelEncoder()
    encoded_df = df_filtered.copy()
    for col in cat_cols:
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        
    # Normalizasyon (MinMax 0-1 aralığı)
    scaler = MinMaxScaler()
    # Sadece sayısal ve encode edilmiş kolonlar
    cols_to_scale = [c for c in encoded_df.columns if c not in ['jail', 'jail_log', 'new_id', 'case_type_id']] 
    # Hata almamak için numeric olanları seç
    cols_to_scale = encoded_df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [c for c in cols_to_scale if c not in ['jail', 'jail_log']]
    
    if cols_to_scale:
        encoded_df[cols_to_scale] = scaler.fit_transform(encoded_df[cols_to_scale])
        print(f"✅ Normalizasyon tamamlandı ({len(cols_to_scale)} kolon).")

    # 4. EN ETKİLİ PARAMETRELERİ BULMA (Korelasyon)
    print("\n🏆 En Etkili Parametre Analizi (Log Hedef ile)...")
    
    corr_matrix = encoded_df.corr()
    target_corr = corr_matrix['jail_log'].abs().sort_values(ascending=False)
    
    print("\n🔝 'jail_log' (Düzeltilmiş Ceza) ile En Yüksek İlişkili 20 Özellik:")
    print(target_corr.head(21)) # jail_log ve jail kendisi dahil
    
    # Feature Importance (Random Forest ile daha sağlam bir seçim)
    from sklearn.ensemble import RandomForestRegressor
    
    # Örneklem al (Hız için 10k)
    sample_df = encoded_df.sample(n=min(10000, len(encoded_df)), random_state=42)
    X = sample_df.drop(['jail', 'jail_log'], axis=1)
    # Hata veren kolonları (object kalmışsa) temizle
    X = X.select_dtypes(include=[np.number])
    y = sample_df['jail_log']
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🌲 Random Forest Feature Importance (En Önemli 20):")
    print(importances.head(20))
    
    importances.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    
    print(f"\n💾 Tüm sonuçlar {OUTPUT_DIR} konumuna kaydedildi.")

if __name__ == "__main__":
    advanced_processing()
