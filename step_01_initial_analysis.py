
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Yollar
VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/new_analysis_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_and_filter():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    print(f"✅ Orijinal Veri Boyutu: {df.shape}")

    # Temel istatistikler (Filtreleme Öncesi)
    if 'jail' in df.columns:
        print("\n📊 Hedef Değişken (jail) Dağılımı (Filtreleme ÖNCESİ):")
        print(df['jail'].describe())
        
        # 0-300 Aralığını Analiz Et
        range_0_300 = df[(df['jail'] >= 0) & (df['jail'] <= 300)]
        print(f"\n⚠️ 0-300 Arası Kayıt Sayısı: {len(range_0_300)} ({len(range_0_300)/len(df)*100:.2f}%)")
        
        # Filtreleme: 0-300 arasını çıkar (Kullanıcının isteği: "0-300 arasını örneklemden çıkaracağız")
        # Mantıken jail > 300 olanları istiyoruz.
        # "300+ ceza tahminlerinde bir miktar başarı sağladık" -> Hedef kitle > 300
        # Ancak 0'a ne yapacağız? Genelde 0 ceza almayanlar demek.
        # Kullanıcı "0-300 arasını örneklemden çıkaracağız" dedi. 
        # Yani jail > 300 OLANLARI tutacağız.
        
        df_filtered = df[df['jail'] > 300].copy()
        print(f"\n✅ Filtrelenmiş Veri (jail > 300): {df_filtered.shape}")
        print(df_filtered['jail'].describe())
    else:
        print("❌ 'jail' kolonu bulunamadı!")
        return

    # Eksik Veri Analizi
    print("\n🔍 Eksik Veri Analizi (Filtrelenmiş Veri Üzerinde):")
    missing = df_filtered.isnull().sum()
    missing_ratio = (missing / len(df_filtered)) * 100
    missing_df = pd.DataFrame({'Missing': missing, 'Ratio': missing_ratio})
    print(missing_df[missing_df['Missing'] > 0].sort_values('Ratio', ascending=False))

    # Yıl ve Hakim Analizi (Eğer kolonlar varsa)
    # Kolon isimlerini tahmin etmeye çalışalım veya hepsini lower yapalım
    df_filtered.columns = [c.lower() for c in df_filtered.columns]
    
    # Normalizasyon ve Korelasyon için hazırlık
    # Sadece sayısal kolonlar
    numeric_df = df_filtered.select_dtypes(include=[np.number])
    
    # Korelasyon
    if 'jail' in numeric_df.columns:
        corr = numeric_df.corr()['jail'].sort_values(ascending=False)
        print("\n📈 'jail' ile En Yüksek Korelasyona Sahip 20 Özellik:")
        print(corr.head(20))
        print("\n📉 'jail' ile En Negatif Korelasyona Sahip 10 Özellik:")
        print(corr.tail(10))
        
        # Kaydet
        corr.to_csv(OUTPUT_DIR / "correlation_jail.csv")

    # Temizlenmiş veriyi kaydet (Örneklem için, çok büyükse sadece bir kısmını veya info'yu)
    # Analiz için sample kaydetmeyelim, rapor üretelim.
    
    print(f"\n💾 Sonuçlar {OUTPUT_DIR} dizinine kaydediliyor...")
    
if __name__ == "__main__":
    analyze_and_filter()
