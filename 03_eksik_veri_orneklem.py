"""
ADIM 3: EKSİK VERİLERDEN %15 ÖRNEKLEM
Kalan 1.1M eksik verili satırlardan rastgele %15 seçme
"""

import pandas as pd
import time

print("=" * 70)
print("ADIM 3: EKSİK VERİLERDEN %15 ÖRNEKLEM ALMA")
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

# Eksik verili satırları ayır
print("\n🔍 Eksik verili satırlar seçiliyor...")
df_missing = df[df.isnull().any(axis=1)]

print(f"✅ Eksik verili satırlar bulundu!")
print(f"📊 Eksik verili satır sayısı: {len(df_missing):,}")
print(f"📊 Eksik veri oranı: %{(len(df_missing) / len(df) * 100):.2f}")

# %15 örneklem al (random_state=42 ile tekrarlanabilir)
print("\n🎲 Rastgele %15 örneklem alınıyor (random_state=42)...")
df_missing_sample = df_missing.sample(frac=0.15, random_state=42)

print(f"✅ Örneklem alındı!")
print(f"📊 Seçilen satır sayısı: {len(df_missing_sample):,}")
print(f"📊 Orijinal eksik verinin %{(len(df_missing_sample) / len(df_missing) * 100):.2f}'i")

# Örneklemi kaydet
output_path = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld_Eksik_Veri_Yuzde15.csv"
print(f"\n💾 Örneklem kaydediliyor: {output_path}")

df_missing_sample.to_csv(output_path, index=False)

print(f"✅ Kayıt tamamlandı!")
print(f"📦 Dosya boyutu: {pd.read_csv(output_path).memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 70)
print("✅ ADIM 3 TAMAMLANDI!")
print("=" * 70)
print(f"\n📌 Sonraki adım: Temiz veri + %15 eksik veri = Final dataset birleştirme")
