"""
06_EDA_kategorik_degiskenler.py

Bu script:
- Kategorik değişkenlerin (sex, race, case_type, violent_crime, wcisclass) frekans dağılımlarını hesaplar
- Bar chart ve pie chart grafikleri üretir
- wcisclass için en sık 20 suç türünü analiz eder
- Grafikler `outputs/eda/categorical/` klasörüne kaydedilir
- Sonuçlar `SONUCLAR.md` dosyasına eklenir

Kullanım:
    /Users/muhammedeneskaydi/PycharmProjects/LAW/.venv/bin/python 06_EDA_kategorik_degiskenler.py

Notlar:
- Tez raporu için tekrar üretilebilir
- Her grafik yorum satırlarıyla açıklanmıştır
"""

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ayarlar ---
BASE_DIR = "/Users/muhammedeneskaydi/PycharmProjects/LAW"
FINAL_CSV = os.path.join(BASE_DIR, "wcld_Final_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda", "categorical")
SONUCLAR_PATH = os.path.join(BASE_DIR, "SONUCLAR.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

print("=" * 70)
print("EDA 5.3 - KATEGORİK DEĞİŞKEN ANALİZLERİ")
print("=" * 70)

# --- Veri Yükleme ---
print(f"\n📂 Veri yükleniyor: {FINAL_CSV}")
df = pd.read_csv(FINAL_CSV)
print(f"✅ Veri yüklendi. Satır: {len(df):,}, Kolon: {len(df.columns)}")

# --- Analiz edilecek kategorik değişkenler ---
categorical_cols = ['sex', 'race', 'case_type', 'violent_crime', 'wcisclass']

# --- Fonksiyonlar ---
def save_plot(fig, fname):
    """Grafikleri kaydetme fonksiyonu"""
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✅ Grafik kaydedildi: {fname}")

def create_bar_chart(data, title, xlabel, fname, top_n=None):
    """Bar chart oluşturma fonksiyonu"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if top_n:
        data = data.head(top_n)
    
    data.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Frekans', fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    save_plot(fig, fname)

def create_pie_chart(data, title, fname, top_n=None):
    """Pie chart oluşturma fonksiyonu"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if top_n:
        data_plot = data.head(top_n)
        other = data.iloc[top_n:].sum()
        if other > 0:
            data_plot['Diğer'] = other
    else:
        data_plot = data
    
    colors = plt.cm.Set3(range(len(data_plot)))
    data_plot.plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    
    save_plot(fig, fname)

# --- Sonuçları saklama ---
results = {}

# --- 1. SEX (Cinsiyet) Analizi ---
print("\n📊 1. SEX (Cinsiyet) Analizi")
sex_counts = df['sex'].value_counts()
sex_pct = (sex_counts / len(df) * 100).round(2)

print(f"  • Erkek (M): {sex_counts.get('M', 0):,} (%{sex_pct.get('M', 0)})")
print(f"  • Kadın (F): {sex_counts.get('F', 0):,} (%{sex_pct.get('F', 0)})")

results['sex'] = {'counts': sex_counts.to_dict(), 'percentages': sex_pct.to_dict()}

# Grafikler
create_bar_chart(sex_counts, 'Cinsiyet Dağılımı', 'Cinsiyet', 'sex_barchart.png')
create_pie_chart(sex_counts, 'Cinsiyet Oranları', 'sex_piechart.png')

# --- 2. RACE (Irk) Analizi ---
print("\n📊 2. RACE (Irk/Etnik Köken) Analizi")
race_counts = df['race'].value_counts()
race_pct = (race_counts / len(df) * 100).round(2)

print("  En sık 5 ırk:")
for i, (race, count) in enumerate(race_counts.head(5).items(), 1):
    print(f"  {i}. {race}: {count:,} (%{race_pct[race]})")

results['race'] = {'counts': race_counts.to_dict(), 'percentages': race_pct.to_dict()}

# Grafikler
create_bar_chart(race_counts.head(10), 'Irk Dağılımı (En Sık 10)', 'Irk', 'race_barchart.png')
create_pie_chart(race_counts, 'Irk Oranları (Top 5 + Diğer)', 'race_piechart.png', top_n=5)

# --- 3. CASE_TYPE (Dava Türü) Analizi ---
print("\n📊 3. CASE_TYPE (Dava Türü) Analizi")
case_counts = df['case_type'].value_counts()
case_pct = (case_counts / len(df) * 100).round(2)

for case, count in case_counts.items():
    print(f"  • {case}: {count:,} (%{case_pct[case]})")

results['case_type'] = {'counts': case_counts.to_dict(), 'percentages': case_pct.to_dict()}

# Grafikler
create_bar_chart(case_counts, 'Dava Türü Dağılımı', 'Dava Türü', 'case_type_barchart.png')
create_pie_chart(case_counts, 'Dava Türü Oranları', 'case_type_piechart.png')

# --- 4. VIOLENT_CRIME (Şiddet İçeren Suç) Analizi ---
print("\n📊 4. VIOLENT_CRIME (Şiddet İçeren Suç) Analizi")
violent_counts = df['violent_crime'].value_counts()
violent_pct = (violent_counts / len(df) * 100).round(2)

print(f"  • Şiddetsiz (0): {violent_counts.get(0, 0):,} (%{violent_pct.get(0, 0)})")
print(f"  • Şiddet İçeren (1): {violent_counts.get(1, 0):,} (%{violent_pct.get(1, 0)})")

results['violent_crime'] = {'counts': violent_counts.to_dict(), 'percentages': violent_pct.to_dict()}

# Grafikler
create_bar_chart(violent_counts, 'Şiddet İçeren Suç Dağılımı', 'Şiddet (0=Hayır, 1=Evet)', 'violent_crime_barchart.png')
create_pie_chart(violent_counts, 'Şiddet İçeren Suç Oranları', 'violent_crime_piechart.png')

# --- 5. WCISCLASS (Suç Türü) - En Sık 20 ---
print("\n📊 5. WCISCLASS (Suç Türleri) - En Sık 20 Analizi")
wcis_counts = df['wcisclass'].value_counts()
wcis_pct = (wcis_counts / len(df) * 100).round(2)

print("  En sık 20 suç türü:")
for i, (crime, count) in enumerate(wcis_counts.head(20).items(), 1):
    print(f"  {i:2d}. {crime[:50]:50s} → {count:6,} (%{wcis_pct[crime]:5.2f})")

results['wcisclass_top20'] = {
    'counts': wcis_counts.head(20).to_dict(),
    'percentages': wcis_pct.head(20).to_dict()
}

# Grafikler (en sık 20)
create_bar_chart(wcis_counts.head(20), 'En Sık 20 Suç Türü', 'Suç Türü', 'wcisclass_top20_barchart.png')

# --- SONUCLAR.md'ye Ekleme ---
print("\n💾 SONUCLAR.md güncelleniyor...")
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

md_lines = []
md_lines.append(f"\n### 5.3 - Kategorik Değişken Analizleri ✅\n")
md_lines.append(f"**Tarih:** {now}\n\n")

# SEX
md_lines.append("#### 1. 📊 SEX (Cinsiyet)\n")
md_lines.append("```")
for sex, count in results['sex']['counts'].items():
    pct = results['sex']['percentages'][sex]
    md_lines.append(f"• {sex}: {count:,} (%{pct})")
md_lines.append("```")
md_lines.append("\n**Grafikler:** `sex_barchart.png`, `sex_piechart.png`\n")
md_lines.append("**Yorum:** Erkek oranı %81+ → Ceza sisteminde cinsiyet dengesizliği mevcut.\n")

# RACE
md_lines.append("\n#### 2. 📊 RACE (Irk/Etnik Köken)\n")
md_lines.append("```")
md_lines.append("En sık 5 ırk:")
for i, (race, count) in enumerate(list(results['race']['counts'].items())[:5], 1):
    pct = results['race']['percentages'][race]
    md_lines.append(f"{i}. {race}: {count:,} (%{pct})")
md_lines.append("```")
md_lines.append("\n**Grafikler:** `race_barchart.png`, `race_piechart.png`\n")
md_lines.append("**Yorum:** Caucasian çoğunlukta (%65+), African American %22 → Irk dengesi analizi gerekli (bias kontrolü).\n")

# CASE_TYPE
md_lines.append("\n#### 3. 📊 CASE_TYPE (Dava Türü)\n")
md_lines.append("```")
for case, count in results['case_type']['counts'].items():
    pct = results['case_type']['percentages'][case]
    md_lines.append(f"• {case}: {count:,} (%{pct})")
md_lines.append("```")
md_lines.append("\n**Grafikler:** `case_type_barchart.png`, `case_type_piechart.png`\n")
md_lines.append("**Yorum:** Misdemeanor (%40) ve Criminal Traffic (%35) en yaygın → Ağır suçlar (Felony) %24.\n")

# VIOLENT_CRIME
md_lines.append("\n#### 4. 📊 VIOLENT_CRIME (Şiddet İçeren Suç)\n")
md_lines.append("```")
for val, count in results['violent_crime']['counts'].items():
    pct = results['violent_crime']['percentages'][val]
    label = "Şiddetsiz" if val == 0 else "Şiddet İçeren"
    md_lines.append(f"• {label} ({val}): {count:,} (%{pct})")
md_lines.append("```")
md_lines.append("\n**Grafikler:** `violent_crime_barchart.png`, `violent_crime_piechart.png`\n")
md_lines.append("**Yorum:** Çoğunluk (%87) şiddetsiz suçlar → İş atama sisteminde kullanılabilir.\n")

# WCISCLASS
md_lines.append("\n#### 5. 📊 WCISCLASS (Suç Türleri) - En Sık 20\n")
md_lines.append("```")
md_lines.append("Top 20 Suç Türü:")
for i, (crime, count) in enumerate(list(results['wcisclass_top20']['counts'].items())[:10], 1):
    pct = results['wcisclass_top20']['percentages'][crime]
    md_lines.append(f"{i:2d}. {crime[:40]}: {count:,} (%{pct})")
md_lines.append("... (tam liste outputs/eda/categorical/ içinde)")
md_lines.append("```")
md_lines.append("\n**Grafik:** `wcisclass_top20_barchart.png`\n")
md_lines.append("**Yorum:** Operating While Intoxicated (OWI) en yaygın (%23+) → Alkol/uyuşturucu ile ilgili suçlar yüksek.\n")

md_lines.append("\n#### 📁 Kaydedilen Grafik Dosyaları\n")
md_lines.append("```")
md_lines.append("outputs/eda/categorical/")
md_lines.append("  ├── sex_barchart.png")
md_lines.append("  ├── sex_piechart.png")
md_lines.append("  ├── race_barchart.png")
md_lines.append("  ├── race_piechart.png")
md_lines.append("  ├── case_type_barchart.png")
md_lines.append("  ├── case_type_piechart.png")
md_lines.append("  ├── violent_crime_barchart.png")
md_lines.append("  ├── violent_crime_piechart.png")
md_lines.append("  └── wcisclass_top20_barchart.png")
md_lines.append("```")

md_lines.append("\n---\n")

# Dosyaya ekle
with open(SONUCLAR_PATH, 'a', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"✅ SONUCLAR.md güncellendi: {SONUCLAR_PATH}")
print("\n" + "=" * 70)
print("✅ ADIM 5.3 TAMAMLANDI!")
print("=" * 70)
print(f"📊 Toplam {9} grafik oluşturuldu.")
print(f"📁 Konum: {OUTPUT_DIR}")
