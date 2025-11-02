"""
TEZ PROJESİ - ADIM 2, 3, 4: Veri Hazırlama Süreci
===================================================
Adım 2: Temiz veri ayırma (tüm kolonları dolu)
Adım 3: Eksik verilerden %15 örneklem
Adım 4: Final veri setini birleştirme
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VERİ HAZIRLIK SÜRECİ - TEZ METODUNA GÖRE")
print("=" * 80)

# Dosya yolları
KAYNAK_DOSYA = "/Users/muhammedeneskaydi/Desktop/3.SINIF 2.DÖNEM/TEZ/TEZ FİNAL/wcld.csv"
CIKTI_KLASOR = "/Users/muhammedeneskaydi/PycharmProjects/LAW/"

print("\n📂 Kaynak veri yükleniyor...")
df = pd.read_csv(KAYNAK_DOSYA)
print(f"✅ Toplam {len(df):,} satır yüklendi")

# ============================================================================
# ADIM 2: TEMİZ VERİ (TÜM KOLONLARI DOLU) AYIRMA
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 2: TEMİZ VERİYİ AYIRMA (Tüm Kolonları Dolu Satırlar)")
print("=" * 80)

df_temiz = df.dropna()
print(f"\n✅ Temiz veri: {len(df_temiz):,} satır (Toplam verinin %{len(df_temiz)/len(df)*100:.2f}'si)")
print(f"   Kolon sayısı: {len(df_temiz.columns)}")

# Temiz veriyi kaydet
temiz_dosya = CIKTI_KLASOR + "wcld_Tüm_Kolonlar_Dolu.csv"
df_temiz.to_csv(temiz_dosya, index=False)
print(f"💾 Temiz veri kaydedildi: {temiz_dosya}")

# ============================================================================
# ADIM 3: EKSİK VERİLERDEN %15 ÖRNEKLEM
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 3: EKSİK VERİLERDEN %15 RASTGELE ÖRNEKLEM")
print("=" * 80)

# Eksik verili satırları ayır
df_eksik = df[df.isnull().any(axis=1)]
print(f"\n📊 Eksik veri: {len(df_eksik):,} satır (Toplam verinin %{len(df_eksik)/len(df)*100:.2f}'si)")

# %15 örneklem
orneklem_orani = 0.15
random_seed = 42
df_eksik_orneklem = df_eksik.sample(frac=orneklem_orani, random_state=random_seed)
print(f"\n✅ %15 örneklem: {len(df_eksik_orneklem):,} satır seçildi")
print(f"   Seçilen satırlar toplam verinin %{len(df_eksik_orneklem)/len(df)*100:.2f}'si")

# ============================================================================
# ADIM 4: FİNAL VERİ SETİNİ BİRLEŞTİRME
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 4: FİNAL VERİ SETİNİ BİRLEŞTİRME")
print("=" * 80)

df_final = pd.concat([df_temiz, df_eksik_orneklem], ignore_index=True)
print(f"\n✅ Final Veri Seti: {len(df_final):,} satır")
print(f"   = Temiz: {len(df_temiz):,}")
print(f"   + Eksik %15: {len(df_eksik_orneklem):,}")
print(f"   = Toplam: {len(df_final):,}")

# Final veriyi kaydet
final_dosya = CIKTI_KLASOR + "wcld_Final_Dataset.csv"
df_final.to_csv(final_dosya, index=False)
print(f"\n💾 Final veri seti kaydedildi: {final_dosya}")

# ============================================================================
# ÖZET BİLGİLER
# ============================================================================
print("\n" + "=" * 80)
print("ÖZET BİLGİLER - VERİ HAZIRLIK SÜRECİ")
print("=" * 80)

ozet_data = {
    "Veri Seti": [
        "1. Orijinal Veri",
        "2. Temiz Veri (Tüm kolonlar dolu)",
        "3. Eksik Veri",
        "4. Eksik Veriden %15 Örneklem",
        "5. ⭐ FİNAL VERİ SETİ"
    ],
    "Satır Sayısı": [
        f"{len(df):,}",
        f"{len(df_temiz):,}",
        f"{len(df_eksik):,}",
        f"{len(df_eksik_orneklem):,}",
        f"{len(df_final):,}"
    ],
    "Oran (%)": [
        f"{100.00:.2f}",
        f"{len(df_temiz)/len(df)*100:.2f}",
        f"{len(df_eksik)/len(df)*100:.2f}",
        f"{len(df_eksik_orneklem)/len(df)*100:.2f}",
        f"{len(df_final)/len(df)*100:.2f}"
    ]
}

ozet_df = pd.DataFrame(ozet_data)
print("\n")
print(ozet_df.to_string(index=False))

# Final verideki eksik değer durumu
print("\n" + "=" * 80)
print("FİNAL VERİ SETİNDE EKSİK DEĞER DURUMU")
print("=" * 80)

eksik_final = df_final.isnull().sum()
eksik_final_oran = (eksik_final / len(df_final) * 100)

eksik_rapor = pd.DataFrame({
    'Kolon': eksik_final.index,
    'Eksik Sayısı': eksik_final.values,
    'Eksik Oran (%)': eksik_final_oran.values
})
eksik_rapor = eksik_rapor[eksik_rapor['Eksik Sayısı'] > 0].sort_values('Eksik Sayısı', ascending=False)

if len(eksik_rapor) > 0:
    print(f"\n⚠️  Final veride eksik değer içeren {len(eksik_rapor)} kolon:")
    print(eksik_rapor.to_string(index=False))
else:
    print("\n✅ Final veride hiç eksik değer yok!")

# Hedef değişkenlerin durumu
print("\n" + "=" * 80)
print("HEDEF DEĞİŞKENLERİN DURUMU (Final Veri Seti)")
print("=" * 80)

hedef_kolonlar = ['jail', 'probation', 'release']
for col in hedef_kolonlar:
    eksik = df_final[col].isnull().sum()
    dolu = df_final[col].notna().sum()
    print(f"\n📊 {col.upper()}:")
    print(f"   Dolu: {dolu:,} ({dolu/len(df_final)*100:.2f}%)")
    print(f"   Eksik: {eksik:,} ({eksik/len(df_final)*100:.2f}%)")
    if dolu > 0:
        print(f"   Min: {df_final[col].min():.2f}, Max: {df_final[col].max():.2f}, Ortalama: {df_final[col].mean():.2f}")

print("\n" + "=" * 80)
print("✅ VERİ HAZIRLIK SÜRECİ TAMAMLANDI!")
print("=" * 80)
print("\n💡 SONRAKI ADIM: Final veri seti üzerinde EDA (Keşifsel Veri Analizi)")
print(f"   Dosya: {final_dosya}")
