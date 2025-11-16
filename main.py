import pandas as pd

# WCLD veri dosyanın yolunu buraya yaz
full_path = "/Users/muhammedeneskaydi/Desktop/wcld.csv"  # eğer aynı dizindeyse sadece dosya adı yeterli

# İlk 10.000 satırı oku
sample_df = pd.read_csv(full_path, nrows=10000)

# Yeni küçük dosyayı kaydet
sample_df.to_csv("/Users/muhammedeneskaydi/Desktop/wcld.YENİ.csv", index=False)