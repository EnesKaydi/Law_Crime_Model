"""
05_EDA_hedef_degisken_dagitimi.py

Bu script:
- `wcld_Final_Dataset.csv` dosyasını yükler
- Hedef değişkenler `jail`, `probation`, `release` için istatistiksel özetler çıkarır
- Histogram ve boxplot grafikleri üretir ve `outputs/eda/target_distributions/` klasörüne kaydeder
- `jail` değerlerine göre ceza kategorisi (Hafif/Orta/Ağır/NoJail) oluşturur ve kategorilere göre dağılım grafiği üretir
- Elde edilen özetleri ve kısa yorumları `SONUCLAR.md` dosyasına ekler (rapor için kullanılacak)

Kullanım:
    /Users/muhammedeneskaydi/PycharmProjects/LAW/.venv/bin/python 05_EDA_hedef_degisken_dagitimi.py

Notlar:
- Script tüm adımları yorum satırlarıyla açıklar; tez raporuna koymak üzere tekrar üretilebilir.
"""

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ayarlar ve yol tanımları -------------------------------------------------
BASE_DIR = "/Users/muhammedeneskaydi/PycharmProjects/LAW"
FINAL_CSV = os.path.join(BASE_DIR, "wcld_Final_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda", "target_distributions")
SONUCLAR_PATH = os.path.join(BASE_DIR, "SONUCLAR.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Görselleştirme stilini ayarla
sns.set(style="whitegrid")

# --- Veri Yükleme ------------------------------------------------------------
print("Veri yükleniyor:", FINAL_CSV)
df = pd.read_csv(FINAL_CSV)
print(f"Veri yüklendi. Satır: {len(df):,}, Kolon: {len(df.columns)}")

# Hedef değişkenler listesi
targets = ["jail", "probation", "release"]

# --- Fonksiyonlar -----------------------------------------------------------
def save_plot(fig, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Grafik kaydedildi: {path}")

# --- İstatistiksel Özet ve Grafik Üretimi -----------------------------------
summary = {"timestamp": datetime.utcnow().isoformat(), "results": {}}

for col in targets:
    col_stats = {}
    if col not in df.columns:
        print(f"Uyarı: {col} bulunamadı, atlanıyor.")
        continue

    s = df[col]
    non_null = s.dropna()

    # Temel istatistikler
    col_stats['count'] = int(s.count())
    col_stats['nulls'] = int(s.isna().sum())
    col_stats['mean'] = float(non_null.mean()) if len(non_null) else None
    col_stats['median'] = float(non_null.median()) if len(non_null) else None
    col_stats['std'] = float(non_null.std()) if len(non_null) else None
    col_stats['min'] = float(non_null.min()) if len(non_null) else None
    col_stats['max'] = float(non_null.max()) if len(non_null) else None
    col_stats['25%'] = float(non_null.quantile(0.25)) if len(non_null) else None
    col_stats['75%'] = float(non_null.quantile(0.75)) if len(non_null) else None

    summary['results'][col] = col_stats

    # Histogram
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(non_null, bins=60, kde=False, ax=ax, color='tab:blue')
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    save_plot(fig, f"hist_{col}.png")

    # Boxplot (outlier kontrolü için)
    fig, ax = plt.subplots(figsize=(6,3))
    sns.boxplot(x=non_null, ax=ax, color='tab:orange')
    ax.set_title(f"Boxplot of {col}")
    save_plot(fig, f"box_{col}.png")

# --- Ceza kategorileri (Hafif/Orta/Ağır) oluşturma --------------------------
# Kurallar: 1-180 => Hafif, 181-1080 => Orta, 1081+ => Ağır. 0 veya NaN => NoJail

def categorize_jail(val):
    try:
        if pd.isna(val):
            return 'NoJail'
        v = float(val)
        if v <= 0:
            return 'NoJail'
        if 1 <= v <= 180:
            return 'Hafif'
        if 181 <= v <= 1080:
            return 'Orta'
        if v >= 1081:
            return 'Agir'
    except Exception:
        return 'Unknown'

print("Ceza kategorileri oluşturuluyor...")
df['ceza_kategori'] = df['jail'].apply(categorize_jail)

cat_counts = df['ceza_kategori'].value_counts(dropna=False)
print(cat_counts)

# Kategori dağılımı grafik
fig, ax = plt.subplots(figsize=(7,4))
cat_counts.plot(kind='bar', color=['#2b8cbe','#7bccc4','#bae4bc','#f0f0f0'], ax=ax)
ax.set_title('Ceza Kategorileri Dağılımı')
ax.set_ylabel('Count')
ax.set_xlabel('Kategori')
save_plot(fig, 'ceza_kategori_barchart.png')

summary['results']['ceza_kategori_counts'] = cat_counts.to_dict()

# --- Özetleri SONUCLAR.md'ye ekleme -----------------------------------------
print('SONUCLAR.md dosyası güncelleniyor...')
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

md_lines = []
md_lines.append(f"## EDA - Hedef Değişken Dağılımları ({now})\n")
for col, stats in summary['results'].items():
    md_lines.append(f"### {col}\n")
    if isinstance(stats, dict):
        for k,v in stats.items():
            md_lines.append(f"- {k}: {v}")
    else:
        md_lines.append(str(stats))
    md_lines.append('\n')

md_lines.append('Grafikler:')
md_lines.append(f"- hist_jail.png, box_jail.png, hist_probation.png, box_probation.png, hist_release.png, box_release.png")
md_lines.append(f"- ceza_kategori_barchart.png\n")
md_lines.append('---\n')

# Eğer SONUCLAR.md yoksa oluştur, varsa ekle
if not os.path.exists(SONUCLAR_PATH):
    with open(SONUCLAR_PATH, 'w', encoding='utf-8') as f:
        f.write('# TEZ PROJESİ SONUÇLARI\n\n')

with open(SONUCLAR_PATH, 'a', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"SONUCLAR.md güncellendi: {SONUCLAR_PATH}")
print('ADIM 5.2 tamamlandı. Grafikler ve özetler outputs/eda/target_distributions içinde.')
