"""
TEZ PROJESİ - ADIM 1: Büyük Veri Setini Yükleme ve İnceleme
============================================================
Amaç: 1.5M satırlık wcld.csv dosyasını okuma ve genel yapı analizi
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADIM 1: VERİ SETİ YÜKLEME VE GENEL YAPISINI İNCELEME")
print("=" * 80)

# Büyük veri dosyasının yolu
DOSYA_YOLU = "/Users/muhammedeneskaydi/Desktop/3.SINIF 2.DÖNEM/TEZ/TEZ FİNAL/wcld.csv"

print("\n📂 Veri seti yükleniyor... (Bu işlem biraz zaman alabilir)")
df = pd.read_csv(DOSYA_YOLU)

print("\n✅ Veri seti başarıyla yüklendi!")
print("=" * 80)

# Temel bilgiler
print("\n📊 VERİ SETİ BOYUT BİLGİLERİ:")
print("-" * 80)
print(f"Toplam Satır Sayısı: {len(df):,}")
print(f"Toplam Kolon Sayısı: {len(df.columns)}")
print(f"Toplam Hücre Sayısı: {df.shape[0] * df.shape[1]:,}")
print(f"Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Kolon bilgileri
print("\n📋 KOLON İSİMLERİ VE TİPLERİ:")
print("-" * 80)
for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
    print(f"{i:2d}. {col:30s} - {dtype}")

# İlk 5 satır
print("\n👁️  İLK 5 SATIR:")
print("-" * 80)
print(df.head())

# Hedef değişkenlerin kontrol edilmesi
print("\n🎯 HEDEF DEĞİŞKENLER (LABELS):")
print("-" * 80)
hedef_degiskenler = ['jail', 'probation', 'release']
for col in hedef_degiskenler:
    if col in df.columns:
        print(f"✓ {col:15s} - Tip: {str(df[col].dtype):10s} - Örnek değerler: {df[col].dropna().head(3).tolist()}")
    else:
        print(f"✗ {col:15s} - BULUNAMADI!")

# Eksik veri analizi
print("\n🔍 EKSİK VERİ ANALİZİ:")
print("-" * 80)
eksik_sayisi = df.isnull().sum()
eksik_oran = (df.isnull().sum() / len(df) * 100)

eksik_df = pd.DataFrame({
    'Kolon': eksik_sayisi.index,
    'Eksik Sayısı': eksik_sayisi.values,
    'Eksik Oran (%)': eksik_oran.values
})
eksik_df = eksik_df[eksik_df['Eksik Sayısı'] > 0].sort_values('Eksik Sayısı', ascending=False)

if len(eksik_df) > 0:
    print(f"\n⚠️  Eksik değer içeren {len(eksik_df)} kolon bulundu:")
    print(eksik_df.to_string(index=False))
else:
    print("\n✅ Hiç eksik değer yok!")

# Tamamen dolu satırlar
tamamen_dolu = df.dropna()
print(f"\n📌 Tüm kolonları DOLU olan satır sayısı: {len(tamamen_dolu):,}")
print(f"   (Toplam verinin %{len(tamamen_dolu)/len(df)*100:.2f}'si)")

eksik_varolan = df[df.isnull().any(axis=1)]
print(f"\n📌 En az 1 eksik değer içeren satır sayısı: {len(eksik_varolan):,}")
print(f"   (Toplam verinin %{len(eksik_varolan)/len(df)*100:.2f}'si)")

# Özet istatistikler
print("\n📈 SAYISAL KOLONLARIN ÖZET İSTATİSTİKLERİ:")
print("-" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("✅ ADIM 1 TAMAMLANDI - Veri seti başarıyla yüklendi ve incelendi!")
print("=" * 80)
print(f"\n💡 SONRAKI ADIM: Temiz veriyi (tüm kolonları dolu) ayırma işlemi")
print(f"   Beklenen temiz veri: ~{len(tamamen_dolu):,} satır")
