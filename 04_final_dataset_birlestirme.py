"""
ADIM 4: FİNAL VERİ SETİ BİRLEŞTİRME
Temiz veri (357K) + Eksik veri %15 (167K) = Final Dataset (~525K)
"""

import pandas as pd
import time

print("=" * 70)
print("ADIM 4: FİNAL VERİ SETİ BİRLEŞTİRME")
print("=" * 70)

# Veri dosyalarının yolları
temiz_veri_yolu = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld_Tüm_Kolonlar_Dolu.csv"
eksik_veri_yolu = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld_Eksik_Veri_Yuzde15.csv"

print("\n📂 Temiz veri yükleniyor...")
start_time = time.time()
df_clean = pd.read_csv(temiz_veri_yolu)
print(f"✅ Temiz veri yüklendi! ({time.time() - start_time:.2f} saniye)")
print(f"📊 Temiz veri satır sayısı: {len(df_clean):,}")

print("\n📂 Eksik veri örneklemi yükleniyor...")
start_time = time.time()
df_missing_sample = pd.read_csv(eksik_veri_yolu)
print(f"✅ Eksik veri yüklendi! ({time.time() - start_time:.2f} saniye)")
print(f"📊 Eksik veri satır sayısı: {len(df_missing_sample):,}")

# Veri setlerini birleştir
print("\n🔗 Veri setleri birleştiriliyor...")
df_final = pd.concat([df_clean, df_missing_sample], ignore_index=True)

print(f"✅ Birleştirme tamamlandı!")
print(f"📊 Final veri seti satır sayısı: {len(df_final):,}")
print(f"📊 Final veri seti kolon sayısı: {len(df_final.columns)}")

# Final veri setini kaydet
output_path = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld_Final_Dataset.csv"
print(f"\n💾 Final veri seti kaydediliyor: {output_path}")

start_time = time.time()
df_final.to_csv(output_path, index=False)
save_time = time.time() - start_time

print(f"✅ Kayıt tamamlandı! ({save_time:.2f} saniye)")
print(f"📦 Dosya boyutu: ~{(len(df_final) * len(df_final.columns) * 8) / 1024**2:.2f} MB (tahmini)")

# Özet bilgiler
print("\n" + "=" * 70)
print("📊 FİNAL VERİ SETİ ÖZETİ")
print("=" * 70)
print(f"✔️ Temiz veri satırları: {len(df_clean):,} (%{len(df_clean)/len(df_final)*100:.2f})")
print(f"✔️ Eksik veri örneklemi: {len(df_missing_sample):,} (%{len(df_missing_sample)/len(df_final)*100:.2f})")
print(f"✔️ Toplam final satır: {len(df_final):,}")
print(f"✔️ Toplam kolon sayısı: {len(df_final.columns)}")

# Hedef değişkenlerin kontrolü
print("\n📊 HEDEF DEĞİŞKENLER (Labels):")
print(f"  • jail (hapis): {df_final['jail'].notna().sum():,} dolu ({df_final['jail'].notna().sum()/len(df_final)*100:.1f}%)")
print(f"  • probation (şartlı tahliye): {df_final['probation'].notna().sum():,} dolu ({df_final['probation'].notna().sum()/len(df_final)*100:.1f}%)")
print(f"  • release (serbest kalma): {df_final['release'].notna().sum():,} dolu ({df_final['release'].notna().sum()/len(df_final)*100:.1f}%)")

print("\n" + "=" * 70)
print("✅ ADIM 4 TAMAMLANDI!")
print("=" * 70)
print(f"\n📌 Sonraki adım: Veri Keşif Analizi (EDA) - Dağılımlar ve Görselleştirmeler")
