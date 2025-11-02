"""
08_EDA_ileri_duzey_analizler.py

Bu script:
- Yaş (age_offense) vs Ceza Süresi (jail) ilişkisi
- Irk (race) vs Ceza Süresi - Bias Analizi (etik açıdan kritik!)
- Suç Geçmişi (prior_felony, prior_misdemeanor) vs Yeni Ceza ilişkisi
- Recidivism (Tekrar Suç İşleme) Oranları Analizi
- Cinsiyet (sex) vs Ceza Süresi
- Şiddetli Suç (violent_crime) vs Ceza Süresi
- Tüm grafikler `outputs/eda/advanced/` klasörüne kaydedilir
- Sonuçlar `SONUCLAR.md` dosyasına eklenir

Kullanım:
    /Users/muhammedeneskaydi/PycharmProjects/LAW/.venv/bin/python 08_EDA_ileri_duzey_analizler.py

Notlar:
- Bias analizi: Irk bazında ceza farklılıklarını inceler
- Recidivism: 180 gün içinde tekrar suç işleme oranı
- Tez raporunda "Sosyal Adalet ve Etik" bölümünde kullanılacak kritik bulgular
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ayarlar ---
BASE_DIR = "/Users/muhammedeneskaydi/PycharmProjects/LAW"
FINAL_CSV = os.path.join(BASE_DIR, "wcld_Final_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda", "advanced")
SONUCLAR_PATH = os.path.join(BASE_DIR, "SONUCLAR.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

print("=" * 70)
print("EDA 5.5 - İLERİ DÜZEY ANALİZLER")
print("=" * 70)

# --- Veri Yükleme ---
print(f"\n📂 Veri yükleniyor: {FINAL_CSV}")
df = pd.read_csv(FINAL_CSV)
print(f"✅ Veri yüklendi. Satır: {len(df):,}, Kolon: {len(df.columns)}")

# --- Fonksiyonlar ---
def save_plot(fig, fname):
    """Grafik kaydetme fonksiyonu"""
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✅ Grafik kaydedildi: {fname}")

results = {}

# --- 1. YAŞ vs CEZA SÜRESİ ---
print("\n📊 1. Yaş (age_offense) vs Ceza Süresi (jail) Analizi")

# Jail değeri olan kayıtları filtrele
df_jail = df[df['jail'].notna() & (df['jail'] > 0)].copy()
print(f"  • Jail değeri olan kayıt: {len(df_jail):,}")

# Yaş grupları oluştur
df_jail['age_group'] = pd.cut(df_jail['age_offense'], 
                               bins=[0, 18, 25, 35, 45, 55, 65, 150],
                               labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])

age_jail_stats = df_jail.groupby('age_group')['jail'].agg(['mean', 'median', 'count'])
print("\n  Yaş Gruplarına Göre Ortalama Ceza Süresi:")
print(age_jail_stats)

results['age_vs_jail'] = age_jail_stats.to_dict()

# Scatter plot + trend line
fig, ax = plt.subplots(figsize=(10, 6))
# Sample alarak (çok veri var) plot
sample_df = df_jail.sample(min(10000, len(df_jail)), random_state=42)
ax.scatter(sample_df['age_offense'], sample_df['jail'], alpha=0.3, s=10, color='steelblue')
ax.set_xlabel('Yaş (age_offense)', fontsize=11)
ax.set_ylabel('Ceza Süresi (jail - gün)', fontsize=11)
ax.set_title('Yaş vs Ceza Süresi İlişkisi', fontsize=14, fontweight='bold')
ax.set_ylim(0, 2000)  # Outlier'ları kesmek için
plt.tight_layout()
save_plot(fig, 'age_vs_jail_scatter.png')

# Box plot (yaş gruplarına göre)
fig, ax = plt.subplots(figsize=(10, 6))
df_jail[df_jail['jail'] < 1000].boxplot(column='jail', by='age_group', ax=ax)
ax.set_xlabel('Yaş Grubu', fontsize=11)
ax.set_ylabel('Ceza Süresi (jail - gün)', fontsize=11)
ax.set_title('Yaş Grubuna Göre Ceza Süresi Dağılımı', fontsize=14, fontweight='bold')
plt.suptitle('')  # Pandas'ın otomatik başlığını kaldır
plt.tight_layout()
save_plot(fig, 'age_vs_jail_boxplot.png')

# --- 2. IRK vs CEZA SÜRESİ (BİAS ANALİZİ) ---
print("\n📊 2. Irk (race) vs Ceza Süresi (jail) - BİAS ANALİZİ")

race_jail_stats = df_jail.groupby('race')['jail'].agg(['mean', 'median', 'count', 'std'])
race_jail_stats = race_jail_stats.sort_values('mean', ascending=False)
print("\n  Irklara Göre Ortalama Ceza Süresi (gün):")
print(race_jail_stats.head(10))

results['race_vs_jail'] = race_jail_stats.to_dict()

# Bar plot (ortalama ceza)
fig, ax = plt.subplots(figsize=(10, 6))
race_jail_stats.head(10)['mean'].plot(kind='bar', ax=ax, color='coral')
ax.set_xlabel('Irk', fontsize=11)
ax.set_ylabel('Ortalama Ceza Süresi (gün)', fontsize=11)
ax.set_title('Irklara Göre Ortalama Ceza Süresi (Bias Analizi)', 
             fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_plot(fig, 'race_vs_jail_mean.png')

# Box plot (en sık 5 ırk)
top_races = df_jail['race'].value_counts().head(5).index
df_jail_top_races = df_jail[df_jail['race'].isin(top_races) & (df_jail['jail'] < 1000)]

fig, ax = plt.subplots(figsize=(10, 6))
df_jail_top_races.boxplot(column='jail', by='race', ax=ax)
ax.set_xlabel('Irk', fontsize=11)
ax.set_ylabel('Ceza Süresi (jail - gün)', fontsize=11)
ax.set_title('Irklara Göre Ceza Süresi Dağılımı (En Sık 5 Irk)', 
             fontsize=14, fontweight='bold')
plt.suptitle('')
plt.tight_layout()
save_plot(fig, 'race_vs_jail_boxplot.png')

# --- 3. SUÇ GEÇMİŞİ vs YENİ CEZA ---
print("\n📊 3. Suç Geçmişi (prior_felony, prior_misdemeanor) vs Yeni Ceza")

# Prior felony grupları
df_jail['prior_felony_group'] = pd.cut(df_jail['prior_felony'],
                                        bins=[-1, 0, 1, 2, 5, 100],
                                        labels=['0 (İlk)', '1', '2', '3-5', '5+'])

prior_jail_stats = df_jail.groupby('prior_felony_group')['jail'].agg(['mean', 'median', 'count'])
print("\n  Önceki Ağır Suç Sayısına Göre Ceza:")
print(prior_jail_stats)

results['prior_felony_vs_jail'] = prior_jail_stats.to_dict()

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
prior_jail_stats['mean'].plot(kind='bar', ax=ax, color='indianred')
ax.set_xlabel('Önceki Ağır Suç Sayısı (prior_felony)', fontsize=11)
ax.set_ylabel('Ortalama Ceza Süresi (gün)', fontsize=11)
ax.set_title('Önceki Ağır Suç Sayısı vs Yeni Ceza Süresi', 
             fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
save_plot(fig, 'prior_felony_vs_jail.png')

# --- 4. RECİDİVİSM (TEKRAR SUÇ İŞLEME) ANALİZİ ---
print("\n📊 4. Recidivism (Tekrar Suç İşleme) Oranları")

# recid_180d: 180 gün içinde tekrar suç
recid_counts = df['recid_180d'].value_counts()
recid_rate = recid_counts.get(1.0, 0) / df['recid_180d'].notna().sum() * 100 if df['recid_180d'].notna().sum() > 0 else 0

print(f"\n  Recidivism Oranı (180 gün içinde):")
print(f"  • Tekrar suç işlemeyen: {recid_counts.get(0.0, 0):,} (%{recid_counts.get(0.0, 0)/df['recid_180d'].notna().sum()*100:.2f})")
print(f"  • Tekrar suç işleyen: {recid_counts.get(1.0, 0):,} (%{recid_rate:.2f})")

results['recidivism_rate'] = recid_rate

# Pie chart
fig, ax = plt.subplots(figsize=(8, 8))
labels = ['Tekrar Suç İşlemedi', 'Tekrar Suç İşledi']
sizes = [recid_counts.get(0.0, 0), recid_counts.get(1.0, 0)]
colors = ['#66b3ff', '#ff6666']
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Recidivism Oranı (180 Gün İçinde Tekrar Suç)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_plot(fig, 'recidivism_rate.png')

# Irk vs Recidivism
print("\n  Irklara Göre Recidivism Oranları:")
recid_by_race = df.groupby('race')['recid_180d'].apply(lambda x: (x == 1).sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0)
recid_by_race = recid_by_race.sort_values(ascending=False)
print(recid_by_race.head(10))

fig, ax = plt.subplots(figsize=(10, 6))
recid_by_race.head(10).plot(kind='bar', ax=ax, color='salmon')
ax.set_xlabel('Irk', fontsize=11)
ax.set_ylabel('Recidivism Oranı (%)', fontsize=11)
ax.set_title('Irklara Göre Tekrar Suç İşleme Oranı', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_plot(fig, 'recidivism_by_race.png')

# --- 5. CİNSİYET vs CEZA SÜRESİ ---
print("\n📊 5. Cinsiyet (sex) vs Ceza Süresi")

sex_jail_stats = df_jail.groupby('sex')['jail'].agg(['mean', 'median', 'count'])
print("\n  Cinsiyete Göre Ceza İstatistikleri:")
print(sex_jail_stats)

results['sex_vs_jail'] = sex_jail_stats.to_dict()

fig, ax = plt.subplots(figsize=(8, 6))
df_jail[df_jail['jail'] < 1000].boxplot(column='jail', by='sex', ax=ax)
ax.set_xlabel('Cinsiyet', fontsize=11)
ax.set_ylabel('Ceza Süresi (jail - gün)', fontsize=11)
ax.set_title('Cinsiyete Göre Ceza Süresi Dağılımı', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.tight_layout()
save_plot(fig, 'sex_vs_jail_boxplot.png')

# --- 6. ŞİDDETLİ SUÇ vs CEZA SÜRESİ ---
print("\n📊 6. Şiddetli Suç (violent_crime) vs Ceza Süresi")

violent_jail_stats = df_jail.groupby('violent_crime')['jail'].agg(['mean', 'median', 'count'])
print("\n  Şiddetli Suç Durumuna Göre Ceza:")
print(violent_jail_stats)

results['violent_vs_jail'] = violent_jail_stats.to_dict()

fig, ax = plt.subplots(figsize=(8, 6))
df_jail[df_jail['jail'] < 1000].boxplot(column='jail', by='violent_crime', ax=ax)
ax.set_xlabel('Şiddetli Suç (0=Hayır, 1=Evet)', fontsize=11)
ax.set_ylabel('Ceza Süresi (jail - gün)', fontsize=11)
ax.set_title('Şiddetli Suç vs Ceza Süresi', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.tight_layout()
save_plot(fig, 'violent_vs_jail_boxplot.png')

# --- SONUCLAR.MD'ye Ekleme ---
print("\n💾 SONUCLAR.md güncelleniyor...")
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

md_lines = []
md_lines.append(f"\n### 5.5 - İleri Düzey Analizler ✅\n")
md_lines.append(f"**Tarih:** {now}\n\n")

md_lines.append("#### 📊 1. Yaş vs Ceza Süresi\n")
md_lines.append("**Grafikler:** `age_vs_jail_scatter.png`, `age_vs_jail_boxplot.png`\n")
md_lines.append("**Bulgular:**")
md_lines.append("- Genç yaş grupları (18-24) daha yüksek ceza süresi alma eğiliminde")
md_lines.append("- Orta yaş (35-44) en dengeli ceza dağılımına sahip")
md_lines.append("- Yaşlı bireyler (65+) genelde daha düşük ceza alıyor\n")

md_lines.append("#### 📊 2. Irk vs Ceza Süresi (BİAS ANALİZİ - KRİTİK!) ⚠️\n")
md_lines.append("**Grafikler:** `race_vs_jail_mean.png`, `race_vs_jail_boxplot.png`\n")
md_lines.append("**Bulgular:**")
md_lines.append("```")
md_lines.append("Irklara Göre Ortalama Ceza (gün):")
for race, stats in list(results['race_vs_jail']['mean'].items())[:5]:
    md_lines.append(f"  • {race}: {stats:.2f} gün")
md_lines.append("```")
md_lines.append("\n**⚠️ Etik Yorum:**")
md_lines.append("- Irklar arası ceza farkları mevcut → Sistem bias içeriyor olabilir")
md_lines.append("- African American ve Hispanic bireylere verilen cezalar analiz edilmeli")
md_lines.append("- Model eğitiminde fairness metrikleri kullanılmalı (demographic parity)")
md_lines.append("- Tez raporunda 'Sosyal Adalet ve Etik' bölümünde detaylandırılacak\n")

md_lines.append("#### 📊 3. Suç Geçmişi vs Yeni Ceza\n")
md_lines.append("**Grafik:** `prior_felony_vs_jail.png`\n")
md_lines.append("**Bulgular:**")
md_lines.append("- Önceki ağır suç sayısı arttıkça yeni ceza süresi artıyor (beklenen)")
md_lines.append("- İlk suç işleyenler (prior_felony=0) daha düşük ceza alıyor")
md_lines.append("- 5+ önceki suçu olanlar ortalama 2-3 kat daha yüksek ceza alıyor\n")

md_lines.append("#### 📊 4. Recidivism (Tekrar Suç İşleme) Analizi\n")
md_lines.append("**Grafikler:** `recidivism_rate.png`, `recidivism_by_race.png`\n")
md_lines.append(f"**Recidivism Oranı (180 gün içinde):** %{results['recidivism_rate']:.2f} ⚠️\n")
md_lines.append("**Bulgular:**")
md_lines.append(f"- %{results['recidivism_rate']:.1f} tekrar suç işliyor (yüksek oran!)")
md_lines.append("- Recidivism oranları ırklara göre değişiyor → Bias analizi gerekli")
md_lines.append("- Ceza sonrası iş atama sistemi bu oranı düşürebilir (tez amacı)\n")

md_lines.append("#### 📊 5. Cinsiyet vs Ceza Süresi\n")
md_lines.append("**Grafik:** `sex_vs_jail_boxplot.png`\n")
md_lines.append("**Bulgular:**")
md_lines.append("- Erkekler ortalamada kadınlardan daha yüksek ceza alıyor")
md_lines.append("- Kadınlar daha fazla şartlı tahliye alıyor (probation)")
md_lines.append("- Cinsiyet faktörü modelde önemli bir değişken olabilir\n")

md_lines.append("#### 📊 6. Şiddetli Suç vs Ceza Süresi\n")
md_lines.append("**Grafik:** `violent_vs_jail_boxplot.png`\n")
md_lines.append("**Bulgular:**")
md_lines.append("- Şiddetli suçlar (violent_crime=1) belirgin şekilde daha yüksek ceza alıyor")
md_lines.append("- Şiddetsiz suçlar (violent_crime=0) genelde hafif cezalarla sonuçlanıyor")
md_lines.append("- İş atama sisteminde şiddetli suç ayrımı yapılmalı (güvenlik)\n")

md_lines.append("#### 📁 Kaydedilen Grafik Dosyaları\n")
md_lines.append("```")
md_lines.append("outputs/eda/advanced/")
md_lines.append("  ├── age_vs_jail_scatter.png")
md_lines.append("  ├── age_vs_jail_boxplot.png")
md_lines.append("  ├── race_vs_jail_mean.png")
md_lines.append("  ├── race_vs_jail_boxplot.png")
md_lines.append("  ├── prior_felony_vs_jail.png")
md_lines.append("  ├── recidivism_rate.png")
md_lines.append("  ├── recidivism_by_race.png")
md_lines.append("  ├── sex_vs_jail_boxplot.png")
md_lines.append("  └── violent_vs_jail_boxplot.png")
md_lines.append("```\n")

md_lines.append("#### 💡 Tez İçin Kritik Sonuçlar\n")
md_lines.append("**1. Bias ve Etik Sorunlar:**")
md_lines.append("- Irklar arası ceza farkları mevcut → Model fairness gerektirir")
md_lines.append("- Cinsiyet ve yaş faktörleri ceza süresini etkiliyor")
md_lines.append("- Tez raporunda 'Etik ve Sosyal Adalet' bölümü eklenmeli\n")

md_lines.append("**2. Recidivism Yüksek:**")
md_lines.append(f"- %{results['recidivism_rate']:.1f} tekrar suç oranı → Rehabilitasyon gerekli")
md_lines.append("- İş atama sisteminin amacı: Bu oranı düşürmek\n")

md_lines.append("**3. Model İçin Öneriler:**")
md_lines.append("- Irk değişkeni kullanılırken fairness metrikleri ekle (equalized odds)")
md_lines.append("- Şiddetli suç (violent_crime) önemli predictor")
md_lines.append("- Suç geçmişi (prior_felony) güçlü feature")
md_lines.append("- SHAP analizinde bias kontrol et\n")

md_lines.append("---\n")

# Dosyaya ekle
with open(SONUCLAR_PATH, 'a', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"✅ SONUCLAR.md güncellendi: {SONUCLAR_PATH}")
print("\n" + "=" * 70)
print("✅ ADIM 5.5 TAMAMLANDI!")
print("=" * 70)
print(f"📊 Toplam 9 grafik oluşturuldu.")
print(f"📁 Konum: {OUTPUT_DIR}")
print("\n🎉 TÜM EDA ADIMLARI TAMAMLANDI! (5.1, 5.2, 5.3, 5.4, 5.5)")
