"""
ADIM 2: TEMİZ VERİ AYIRMA
Tüm kolonları dolu olan satırları seçip kaydetme
"""

import pandas as pd
import time

print("=" * 70)
print("ADIM 2: TEMİZ VERİ AYIRMA - TÜM KOLONLAR DOLU")
print("=" * 70)

# Veri dosyasının yolu
veri_yolu = "/Users/muhammedeneskaydi/Desktop/3.SINIF 2.DÖNEM/TEZ/TEZ FİNAL/wcld.csv"

print("\n📂 Veri yükleniyor...")
start_time = time.time()

# Büyük veriyi oku
df = pd.read_csv(veri_yolu)

load_time = time.time() - start_time
print(f"✅ Veri yüklendi! ({load_time:.2f} saniye)")
print(f"📊 Toplam satır sayısı: {len(df):,}")

# Temiz veriyi ayır (tüm kolonlar dolu)
print("\n🔍 Tüm kolonları dolu olan satırlar seçiliyor...")
df_clean = df.dropna()

print(f"✅ Temiz veri seçildi!")
print(f"📊 Temiz satır sayısı: {len(df_clean):,}")
print(f"📊 Temiz veri oranı: %{(len(df_clean) / len(df) * 100):.2f}")

# Temiz veriyi kaydet
output_path = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld_Tüm_Kolonlar_Dolu.csv"
print(f"\n💾 Temiz veri kaydediliyor: {output_path}")

df_clean.to_csv(output_path, index=False)

print(f"✅ Kayıt tamamlandı!")
print(f"📦 Dosya boyutu: {pd.read_csv(output_path).memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 70)
print("✅ ADIM 2 TAMAMLANDI!")
print("=" * 70)
print(f"\n📌 Sonraki adım: Eksik verilerden %15 örneklem alma")
