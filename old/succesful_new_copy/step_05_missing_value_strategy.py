
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("../outputs/advanced_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_missing_and_bias():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    # Filtreleme (Önceki kararlarımız)
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # 1. EKSİK VERİ ANALİZİ (RECIDIVISM)
    print("\n🔍 Eksik Veri Analizi: 'is_recid_new' ve Türevleri")
    
    recid_cols = ['is_recid_new', 'recid_180d', 'recid_180d_violent']
    
    for col in recid_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            print(f"\nKolon: {col}")
            print(f"  • Boş Sayısı: {null_count} (%{null_count/len(df)*100:.2f})")
            
            # Hipotez: Boş olması 'Hayır' (0) anlamına mı geliyor?
            # Boş olanların 'jail' ortalaması ile Dolu olanlarınkini kıyaslayalım
            mean_null = df[df[col].isnull()]['jail'].mean()
            mean_not_null = df[df[col].notnull()]['jail'].mean()
            
            print(f"  • Boş Olanların Ort. Cezası: {mean_null:.2f} gün")
            print(f"  • Dolu Olanların Ort. Cezası: {mean_not_null:.2f} gün")
            
            if mean_null < mean_not_null:
                print("  👉 YORUM: Boş olanlar daha az ceza alıyor, muhtemelen 'Suçsuz/Tekrar Yok' demek.")
            else:
                print("  👉 YORUM: Boş olanlar daha çok ceza alıyor, veri kaybı olabilir.")

    # 2. HAKİM ETKİSİ (JUDGE BIAS) SİMÜLASYONU
    print("\n⚖️ Hakim Etkisi Analizi (Simülasyon İçin)")
    
    if 'judge_id' in df.columns:
        # Hakimlerin ortalama cezası ve global ortalamadan farkı
        global_mean = df['jail'].mean()
        judge_stats = df.groupby('judge_id')['jail'].agg(['mean', 'count', 'std'])
        
        # Sadece yeterli davası olan hakimleri alalım (Güvenilirlik için)
        judge_stats = judge_stats[judge_stats['count'] > 20]
        
        judge_stats['bias_days'] = judge_stats['mean'] - global_mean
        judge_stats['bias_percent'] = (judge_stats['bias_days'] / global_mean) * 100
        
        print(f"  • Global Ortalama Ceza: {global_mean:.2f} gün")
        print(f"  • En Sert 5 Hakim (Ortalamanın Üzerinde):")
        print(judge_stats.sort_values('bias_percent', ascending=False).head(5)[['mean', 'bias_days', 'bias_percent']])
        
        print(f"  • En Yumuşak 5 Hakim (Ortalamanın Altında):")
        print(judge_stats.sort_values('bias_percent', ascending=True).head(5)[['mean', 'bias_days', 'bias_percent']])
        
        # Bu tabloyu pickle/csv yap ki web uygulamasında kullanabilsin
        judge_stats.to_csv(OUTPUT_DIR / "judge_bias_map.csv")
        print(f"✅ Hakim bias haritası kaydedildi: {OUTPUT_DIR}/judge_bias_map.csv")
        
        # Görselleştirme
        plt.figure(figsize=(10, 6))
        sns.histplot(judge_stats['bias_percent'], kde=True)
        plt.title('Hakimlerin Ceza Verme Eğilimi (Ortalamaya Göre % Fark)')
        plt.xlabel('% Fark (Pozitif = Sert, Negatif = Yumuşak)')
        plt.axvline(0, color='red', linestyle='--')
        plt.savefig(OUTPUT_DIR / "judge_bias_distribution.png")
        plt.close()

if __name__ == "__main__":
    analyze_missing_and_bias()
