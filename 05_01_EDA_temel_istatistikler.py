"""
=============================================================================
ADIM 5.1: VERİ KEŞİF ANALİZİ (EDA) - TEMEL İSTATİSTİKLER
=============================================================================

Bu script, final veri seti üzerinde temel istatistiksel analizler yapar:
1. Veri boyutu ve yapısı
2. Veri tipleri (kategorik/sayısal)
3. Eksik değer analizi (her kolon için detaylı)
4. Sayısal değişkenlerin özet istatistikleri (mean, median, std, min, max)
5. Kategorik değişkenlerin benzersiz değer sayıları

Çıktılar:
- Konsol'da detaylı raporlar
- outputs/temel_istatistikler.txt dosyasına kayıt

Yazar: Muhammed Enes Kaydı
Tarih: 2 Kasım 2025
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADIM 5.1: VERİ KEŞİF ANALİZİ (EDA) - TEMEL İSTATİSTİKLER")
print("=" * 80)

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================
print("\n[1/6] 📂 Final veri seti yükleniyor...")
veri_yolu = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld_Final_Dataset.csv"

try:
    df = pd.read_csv(veri_yolu)
    print(f"✅ Veri başarıyla yüklendi!")
except FileNotFoundError:
    print("❌ HATA: Veri dosyası bulunamadı!")
    print(f"   Aranan dosya: {veri_yolu}")
    print("   Lütfen önce ADIM 4'ü (final_dataset_birlestirme.py) çalıştırın.")
    exit(1)

# ============================================================================
# 2. VERİ BOYUTU VE YAPISI
# ============================================================================
print("\n[2/6] 📊 Veri boyutu ve yapısı analiz ediliyor...")
print("\n" + "─" * 80)
print("VERİ SETİ GENEL BİLGİLERİ")
print("─" * 80)
print(f"📏 Satır Sayısı (Örnek): {len(df):,}")
print(f"📏 Kolon Sayısı (Özellik): {len(df.columns)}")
print(f"💾 Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"📦 Veri Seti Boyutu: {len(df) * len(df.columns):,} hücre")

# ============================================================================
# 3. VERİ TİPLERİ ANALİZİ
# ============================================================================
print("\n[3/6] 🔤 Veri tipleri analiz ediliyor...")
print("\n" + "─" * 80)
print("VERİ TİPLERİ DAĞILIMI")
print("─" * 80)

# Veri tiplerini say
veri_tipleri = df.dtypes.value_counts()
print("\n📊 Kolon Tipleri:")
for tip, sayi in veri_tipleri.items():
    print(f"   • {tip}: {sayi} kolon")

# Kategorik ve sayısal kolonları ayır
sayisal_kolonlar = df.select_dtypes(include=[np.number]).columns.tolist()
kategorik_kolonlar = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n✔️ Sayısal Kolonlar: {len(sayisal_kolonlar)} adet")
print(f"✔️ Kategorik Kolonlar: {len(kategorik_kolonlar)} adet")

# ============================================================================
# 4. EKSİK DEĞER ANALİZİ (DETAYLI)
# ============================================================================
print("\n[4/6] 🔍 Eksik değerler analiz ediliyor...")
print("\n" + "─" * 80)
print("EKSİK DEĞER ANALİZİ")
print("─" * 80)

# Her kolon için eksik değer sayısı
eksik_degerler = df.isnull().sum()
eksik_yuzde = (df.isnull().sum() / len(df)) * 100

# Eksik değer tablosu oluştur
eksik_tablo = pd.DataFrame({
    'Kolon': df.columns,
    'Eksik Sayı': eksik_degerler.values,
    'Eksik %': eksik_yuzde.values
})

# Sadece eksik değeri olan kolonları göster
eksik_tablo = eksik_tablo[eksik_tablo['Eksik Sayı'] > 0].sort_values('Eksik Sayı', ascending=False)

if len(eksik_tablo) > 0:
    print(f"\n⚠️ Eksik değer içeren kolon sayısı: {len(eksik_tablo)}")
    print(f"\n📋 İLK 20 KOLON (En Çok Eksik Değer İçeren):\n")
    print(eksik_tablo.head(20).to_string(index=False))
    
    # Toplam eksik değer istatistikleri
    print(f"\n📊 TOPLAM EKSİK DEĞER İSTATİSTİKLERİ:")
    print(f"   • Toplam eksik hücre: {eksik_degerler.sum():,}")
    print(f"   • Genel eksik oran: %{(eksik_degerler.sum() / (len(df) * len(df.columns)) * 100):.2f}")
else:
    print("\n✅ Hiç eksik değer yok! Tüm hücreler dolu.")

# ============================================================================
# 5. SAYISAL DEĞİŞKENLER - ÖZET İSTATİSTİKLER
# ============================================================================
print("\n[5/6] 📈 Sayısal değişkenler için özet istatistikler hesaplanıyor...")
print("\n" + "─" * 80)
print("SAYISAL DEĞİŞKENLER - ÖZET İSTATİSTİKLER")
print("─" * 80)

# Hedef değişkenler (labels)
hedef_degiskenler = ['jail', 'probation', 'release']

print("\n🎯 HEDEF DEĞİŞKENLER (Labels):")
print("─" * 80)
for hedef in hedef_degiskenler:
    if hedef in df.columns:
        print(f"\n📌 {hedef.upper()}:")
        print(f"   • Dolu: {df[hedef].notna().sum():,} (%{df[hedef].notna().sum()/len(df)*100:.1f})")
        print(f"   • Eksik: {df[hedef].isna().sum():,} (%{df[hedef].isna().sum()/len(df)*100:.1f})")
        
        # Sayısal istatistikler (sadece dolu değerler için)
        if df[hedef].notna().sum() > 0:
            print(f"   • Ortalama: {df[hedef].mean():.2f}")
            print(f"   • Medyan: {df[hedef].median():.2f}")
            print(f"   • Std. Sapma: {df[hedef].std():.2f}")
            print(f"   • Min: {df[hedef].min():.2f}")
            print(f"   • Max: {df[hedef].max():.2f}")
            print(f"   • Q1 (25%): {df[hedef].quantile(0.25):.2f}")
            print(f"   • Q3 (75%): {df[hedef].quantile(0.75):.2f}")

print("\n\n📊 DİĞER ÖNEMLİ SAYISAL DEĞİŞKENLER:")
print("─" * 80)

# Diğer önemli sayısal değişkenler
onemli_sayisal = ['age_offense', 'prior_felony', 'prior_misdemeanor', 
                   'prior_criminal_traffic', 'violent_crime', 'recid_180d']

for kolon in onemli_sayisal:
    if kolon in df.columns:
        print(f"\n📌 {kolon.upper()}:")
        print(f"   • Ortalama: {df[kolon].mean():.2f}")
        print(f"   • Medyan: {df[kolon].median():.2f}")
        print(f"   • Min: {df[kolon].min():.2f}")
        print(f"   • Max: {df[kolon].max():.2f}")

# ============================================================================
# 6. KATEGORİK DEĞİŞKENLER - BENZERSİZ DEĞERLER
# ============================================================================
print("\n[6/6] 🏷️ Kategorik değişkenler analiz ediliyor...")
print("\n" + "─" * 80)
print("KATEGORİK DEĞİŞKENLER - BENZERSİZ DEĞER SAYILARI")
print("─" * 80)

onemli_kategorik = ['sex', 'race', 'case_type', 'wcisclass', 'all_races']

for kolon in onemli_kategorik:
    if kolon in df.columns:
        benzersiz = df[kolon].nunique()
        print(f"\n📌 {kolon.upper()}:")
        print(f"   • Benzersiz değer sayısı: {benzersiz}")
        
        # En sık 5 değer
        if benzersiz <= 10:  # Eğer az değer varsa hepsini göster
            print(f"   • Değerler ve frekanslar:")
            for deger, sayi in df[kolon].value_counts().items():
                print(f"      - {deger}: {sayi:,} (%{sayi/len(df)*100:.1f})")
        else:  # Çok değer varsa sadece ilk 5'i
            print(f"   • En sık 5 değer:")
            for deger, sayi in df[kolon].value_counts().head(5).items():
                print(f"      - {deger}: {sayi:,} (%{sayi/len(df)*100:.1f})")

# ============================================================================
# 7. SONUÇLARI DOSYAYA KAYDET
# ============================================================================
print("\n" + "=" * 80)
print("💾 Sonuçlar kaydediliyor...")

output_path = "/Users/muhammedeneskaydi/PycharmProjects/LAW/outputs/temel_istatistikler.txt"

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TEZ PROJESİ - TEMEL İSTATİSTİKLER RAPORU\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Tarih: 2 Kasım 2025\n")
    f.write(f"Veri Seti: wcld_Final_Dataset.csv\n\n")
    
    f.write("VERİ SETİ GENEL BİLGİLERİ\n")
    f.write("─" * 80 + "\n")
    f.write(f"Satır Sayısı: {len(df):,}\n")
    f.write(f"Kolon Sayısı: {len(df.columns)}\n")
    f.write(f"Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
    
    f.write("VERİ TİPLERİ\n")
    f.write("─" * 80 + "\n")
    f.write(f"Sayısal Kolonlar: {len(sayisal_kolonlar)}\n")
    f.write(f"Kategorik Kolonlar: {len(kategorik_kolonlar)}\n\n")
    
    f.write("EKSİK DEĞER ANALİZİ\n")
    f.write("─" * 80 + "\n")
    if len(eksik_tablo) > 0:
        f.write(eksik_tablo.to_string(index=False))
    else:
        f.write("Hiç eksik değer yok!\n")
    
    f.write("\n\nHEDEF DEĞİŞKENLER İSTATİSTİKLERİ\n")
    f.write("─" * 80 + "\n")
    for hedef in hedef_degiskenler:
        if hedef in df.columns:
            f.write(f"\n{hedef.upper()}:\n")
            f.write(df[hedef].describe().to_string())
            f.write("\n")

print(f"✅ Sonuçlar kaydedildi: {output_path}")

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "=" * 80)
print("✅ ADIM 5.1 TAMAMLANDI!")
print("=" * 80)
print(f"\n📊 Analiz Edilen Veri: {len(df):,} satır × {len(df.columns)} kolon")
print(f"📁 Çıktı Dosyası: outputs/temel_istatistikler.txt")
print(f"\n📌 Sonraki Adım: ADIM 5.2 - Hedef Değişken Dağılımları")
print(f"   Çalıştır: python 05_02_EDA_hedef_degiskenler.py")
print("=" * 80)
