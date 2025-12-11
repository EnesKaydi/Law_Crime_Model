
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/clustering_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_clustering():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Filtreleme (Mainstream + High Severity hepsi dahil olsun ki genel tabloyu görelim)
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    print(f"✅ Analiz Verisi: {len(df)} satır")
    
    # Kümeleme için Özellikler (Kişi ve Suç Odaklı)
    # Ceza (jail) buraya dahil edilmez, sonuçtur. Biz "Tip" arıyoruz.
    cluster_features = [
        'age_offense',       # Yaş
        'highest_severity',  # Suçun Ağırlığı
        'prior_felony',      # Ağır Sabıka Sayısı
        'prior_misdemeanor', # Hafif Sabıka Sayısı
        'violent_crime',     # Şiddet Var mı?
        'is_recid_new'       # Mükerrer mi?
    ]
    
    # Eksikleri Doldur
    X = df[cluster_features].copy()
    X = X.fillna(X.mean())
    
    # Ölçeklendirme (StandardScaler şart)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means (4 Küme deneyelim: Genç-Hafif, Genç-Ağır, Yaşlı-Hafif, Yaşlı-Ağır gibi)
    print("🔄 K-Means (k=4) çalıştırılıyor...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = clusters
    
    # --- 1. PERSONA ANALİZİ (Cluster Profilleri) ---
    print("\n👥 Küme Profilleri (Ortalamalar):")
    # Her kümenin özellik ortalamaları + Ortalama Ceza (jail)
    stats_cols = cluster_features + ['jail']
    profile = df.groupby('cluster')[stats_cols].mean().round(2)
    
    # Her kümeye isim verelim (Otomatik analiz)
    # Bu kısım dinamik olsa da çıktıya bakıp manuel isimlendirmek daha iyidir.
    # Şimdilik istatistikleri yazalım.
    print(profile)
    
    # Adetler
    counts = df['cluster'].value_counts().sort_index()
    print("\n📊 Küme Büyüklüleri:")
    print(counts)
    
    # --- 2. GÖRSELLEŞTİRME (PCA ile 2 Boyut) ---
    print("\n🎨 PCA Görselleştirme Hazırlanıyor...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis', alpha=0.6, s=50)
    plt.title('Suçlu Profilleri Haritası (K-Means & PCA)')
    plt.xlabel('Bileşen 1 (Genel Suç Profili)')
    plt.ylabel('Bileşen 2 (Detay Ayrışımı)')
    plt.legend(title='Küme (Cluster)')
    plt.savefig(OUTPUT_DIR / "cluster_pca_map.png")
    
    # --- 3. KÜME DETAY GRAFİKLERİ ---
    # Her kümenin Ceza Dağılımı (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='jail', data=df, palette='viridis')
    plt.title('Kümelerin Ceza Dağılımları')
    plt.ylabel('Hapis Cezası (Gün)')
    plt.savefig(OUTPUT_DIR / "cluster_jail_dist.png")
    
    # Her kümenin Şiddet ve Sabıka Durumu
    # Heatmap şeklinde profil özeti
    plt.figure(figsize=(10, 6))
    sns.heatmap(profile.drop(columns=['jail']).T, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Küme Karakteristikleri (Isı Haritası)')
    plt.savefig(OUTPUT_DIR / "cluster_heatmap.png")
    
    print(f"\n💾 Grafikler kaydedildi: {OUTPUT_DIR}")
    
    # Sonuçları Kaydet
    clustering_summary = profile.copy()
    clustering_summary['count'] = counts
    clustering_summary.to_csv(OUTPUT_DIR / "cluster_profiles.csv")
    print("📄 Profil özeti CSV olarak kaydedildi.")

if __name__ == "__main__":
    analyze_clustering()
