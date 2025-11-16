import pandas as pd

# Dosyanın yolunu belirt
file_path = "/Users/muhammedeneskaydi/Desktop/wcld.csv"  # <-- Dosya yolunu buraya yaz, örneğin: "/Users/enes/Desktop/wcld.csv"

# Tüm veriyi oku (büyük dosya olduğu için biraz zaman alabilir)
df = pd.read_csv(file_path)

# Tamamen dolu (hiç NaN olmayan) satırları al
df_clean = df.dropna()

# Yeni tabloyu kaydet (örneğin: "wcld_clean.csv")
df_clean.to_csv("wcld_Tüm_Kolonlar_Dolu.csv", index=False)

# Özet tablo oluştur
summary = {
    "Toplam Satır Sayısı": len(df),
    "Tamamen Dolu Satır Sayısı": len(df_clean),
    "Eksik Verili Satır Sayısı": len(df) - len(df_clean),
    "Dolu Satır Oranı (%)": round((len(df_clean) / len(df)) * 100, 2)
}

# Görsel tablo
summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=["Değer"])
print(summary_df)