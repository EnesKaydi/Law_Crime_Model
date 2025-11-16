import pandas as pd

# 1. Veri setini yükle
df = pd.read_csv("/Users/muhammedeneskaydi/Desktop/wcld.csv")  # Dosya masaüstündeyse yolunu güncelle: "/Users/.../Desktop/wcld.csv"

# 2. Eksiksiz (NaN olmayan) satırları al
df_clean = df.dropna()

# 3. Eksik verili satırları ayır
df_missing = df[df.isnull().any(axis=1)]

# 4. Eksik verili satırlardan %15 rastgele seç
df_missing_sample = df_missing.sample(frac=0.15, random_state=42)

# 5. Final veri setini oluştur
df_final = pd.concat([df_clean, df_missing_sample], ignore_index=True)

# 6. Yeni veri setini kaydet
df_final.to_csv("/Users/muhammedeneskaydi/Desktop/wcld_Temiz+YüzdeONBES1.csv", index=False)

# 7. Özet bilgi
print("✅ Final veri seti oluşturuldu.")
print("✔️ Dolu satır sayısı:", len(df_clean))
print("✔️ Eksiklerden eklenen:", len(df_missing_sample))
print("✔️ Final veri seti toplam satır:", len(df_final))