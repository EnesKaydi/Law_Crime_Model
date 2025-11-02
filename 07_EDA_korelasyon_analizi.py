"""
07_EDA_korelasyon_analizi.py

Bu script:
- Sayısal değişkenler arasındaki korelasyon matrisini hesaplar
- Korelasyon heatmap (ısı haritası) oluşturur
- Hedef değişkenler (jail, probation, release) ile en yüksek korelasyonlu özellikleri bulur
- Multicollinearity (çoklu doğrusallık) kontrolü yapar
- Tüm grafikler `outputs/eda/correlation/` klasörüne kaydedilir
- Sonuçlar `SONUCLAR.md` dosyasına eklenir

Kullanım:
    /Users/muhammedeneskaydi/PycharmProjects/LAW/.venv/bin/python 07_EDA_korelasyon_analizi.py

Notlar:
- Korelasyon katsayısı: -1 ile +1 arası
  * +1: Mükemmel pozitif korelasyon
  * 0: Korelasyon yok
  * -1: Mükemmel negatif korelasyon
- |korelasyon| > 0.7: Güçlü ilişki
- |korelasyon| > 0.9: Multicollinearity riski (model için sorun)
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
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda", "correlation")
SONUCLAR_PATH = os.path.join(BASE_DIR, "SONUCLAR.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="white")

print("=" * 70)
print("EDA 5.4 - KORELASYON ANALİZLERİ")
print("=" * 70)

# --- Veri Yükleme ---
print(f"\n📂 Veri yükleniyor: {FINAL_CSV}")
df = pd.read_csv(FINAL_CSV)
print(f"✅ Veri yüklendi. Satır: {len(df):,}, Kolon: {len(df.columns)}")

# --- Sadece sayısal kolonları seç ---
print("\n🔍 Sayısal kolonlar seçiliyor...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"✅ {len(numeric_cols)} sayısal kolon bulundu")

# Sayısal veri çerçevesi
df_numeric = df[numeric_cols]

# --- Fonksiyonlar ---
def save_plot(fig, fname):
    """Grafik kaydetme fonksiyonu"""
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✅ Grafik kaydedildi: {fname}")

# --- 1. Tam Korelasyon Matrisi Hesaplama ---
print("\n📊 1. Korelasyon matrisi hesaplanıyor...")
corr_matrix = df_numeric.corr()
print(f"✅ Korelasyon matrisi hesaplandı: {corr_matrix.shape}")

# Tam korelasyon matrisi heatmap (tüm değişkenler)
print("  🖼️ Tam korelasyon heatmap oluşturuluyor...")
fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
            linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)
ax.set_title('Korelasyon Matrisi - Tüm Sayısal Değişkenler', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
save_plot(fig, 'correlation_matrix_full.png')

# --- 2. Hedef Değişkenlerle Korelasyonlar ---
print("\n📊 2. Hedef değişkenlerle korelasyonlar analiz ediliyor...")

target_vars = ['jail', 'probation', 'release']
target_correlations = {}

for target in target_vars:
    if target in corr_matrix.columns:
        print(f"\n  🎯 {target.upper()} ile korelasyonlar:")
        
        # Hedef değişkenle korelasyonları al (kendisi hariç)
        corr_with_target = corr_matrix[target].drop(target).sort_values(ascending=False)
        
        # En yüksek pozitif korelasyonlar (top 10)
        top_positive = corr_with_target.head(10)
        print(f"\n    📈 En yüksek POZİTİF korelasyonlar:")
        for i, (feat, val) in enumerate(top_positive.items(), 1):
            print(f"      {i:2d}. {feat:30s} → {val:+.4f}")
        
        # En yüksek negatif korelasyonlar (bottom 10)
        top_negative = corr_with_target.tail(10)
        print(f"\n    📉 En yüksek NEGATİF korelasyonlar:")
        for i, (feat, val) in enumerate(top_negative.items(), 1):
            print(f"      {i:2d}. {feat:30s} → {val:+.4f}")
        
        target_correlations[target] = {
            'positive': top_positive.to_dict(),
            'negative': top_negative.to_dict()
        }
        
        # Hedef değişken korelasyon bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Top 20 (pozitif + negatif)
        top_corr = pd.concat([top_positive.head(10), top_negative.tail(10)])
        colors = ['green' if x > 0 else 'red' for x in top_corr.values]
        
        top_corr.plot(kind='barh', ax=ax, color=colors)
        ax.set_title(f'{target.upper()} ile En Yüksek Korelasyonlar (Top 20)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Korelasyon Katsayısı', fontsize=11)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        save_plot(fig, f'correlation_{target}_top20.png')

# --- 3. Multicollinearity Kontrolü ---
print("\n📊 3. Multicollinearity (Çoklu Doğrusallık) Kontrolü")
print("  ⚠️ |korelasyon| > 0.9 olan çiftler aranıyor...")

# Üst üçgen matris (tekrarları önlemek için)
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Yüksek korelasyonlu çiftleri bul (|r| > 0.9)
high_corr_pairs = []
for col in upper_triangle.columns:
    for idx in upper_triangle.index:
        val = upper_triangle.loc[idx, col]
        if pd.notna(val) and abs(val) > 0.9:
            high_corr_pairs.append({
                'feature_1': idx,
                'feature_2': col,
                'correlation': val
            })

if high_corr_pairs:
    print(f"\n  ⚠️ {len(high_corr_pairs)} adet yüksek korelasyonlu çift bulundu:")
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', 
                                                               ascending=False)
    print(high_corr_df.to_string(index=False))
else:
    print("  ✅ Yüksek korelasyonlu çift bulunamadı (multicollinearity yok)")

# --- 4. Önemli Özellikler için Detaylı Korelasyon ---
print("\n📊 4. Önemli özellikler için detaylı korelasyon heatmap")

# Model için önemli olabilecek özellikler
important_features = [
    'jail', 'probation', 'release',
    'age_offense', 'prior_felony', 'prior_misdemeanor',
    'violent_crime', 'recid_180d', 'is_recid_new',
    'highest_severity', 'max_hist_jail', 'avg_hist_jail',
    'pct_black', 'pct_college', 'med_hhinc'
]

# Mevcut olanları seç
available_features = [f for f in important_features if f in df_numeric.columns]
print(f"  📋 {len(available_features)} önemli özellik seçildi")

if len(available_features) > 1:
    corr_important = df_numeric[available_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_important, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Korelasyon Matrisi - Önemli Özellikler',
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    save_plot(fig, 'correlation_important_features.png')

# --- SONUCLAR.md'ye Ekleme ---
print("\n💾 SONUCLAR.md güncelleniyor...")
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

md_lines = []
md_lines.append(f"\n### 5.4 - Korelasyon Analizleri ✅\n")
md_lines.append(f"**Tarih:** {now}\n\n")
md_lines.append("#### 📊 Genel Bakış\n")
md_lines.append(f"- Toplam sayısal değişken: {len(numeric_cols)}")
md_lines.append(f"- Korelasyon matrisi boyutu: {corr_matrix.shape[0]}x{corr_matrix.shape[1]}")
md_lines.append(f"- Multicollinearity (|r|>0.9): {len(high_corr_pairs)} çift\n")

# Hedef değişkenler için korelasyonlar
for target in target_vars:
    if target in target_correlations:
        md_lines.append(f"\n#### 🎯 {target.upper()} ile En Yüksek Korelasyonlar\n")
        
        md_lines.append("**Pozitif Korelasyonlar (Top 10):**")
        md_lines.append("```")
        for i, (feat, val) in enumerate(list(target_correlations[target]['positive'].items())[:10], 1):
            md_lines.append(f"{i:2d}. {feat:35s} → {val:+.4f}")
        md_lines.append("```\n")
        
        md_lines.append("**Negatif Korelasyonlar (Top 10):**")
        md_lines.append("```")
        for i, (feat, val) in enumerate(list(target_correlations[target]['negative'].items())[:10], 1):
            md_lines.append(f"{i:2d}. {feat:35s} → {val:+.4f}")
        md_lines.append("```\n")
        
        md_lines.append(f"**Grafik:** `correlation_{target}_top20.png`\n")

# Multicollinearity
md_lines.append("\n#### ⚠️ Multicollinearity Kontrolü\n")
if high_corr_pairs:
    md_lines.append(f"**{len(high_corr_pairs)} adet yüksek korelasyonlu çift bulundu (|r| > 0.9):**")
    md_lines.append("```")
    for pair in high_corr_pairs[:10]:  # İlk 10
        md_lines.append(f"• {pair['feature_1']:30s} ↔ {pair['feature_2']:30s} → {pair['correlation']:+.4f}")
    md_lines.append("```")
    md_lines.append("\n**Öneri:** Model eğitiminde bu değişkenlerden birini çıkar (VIF analizi yap).\n")
else:
    md_lines.append("✅ Yüksek korelasyonlu çift bulunamadı. Multicollinearity sorunu yok.\n")

# Grafikler
md_lines.append("\n#### 📁 Kaydedilen Grafik Dosyaları\n")
md_lines.append("```")
md_lines.append("outputs/eda/correlation/")
md_lines.append("  ├── correlation_matrix_full.png (Tam korelasyon matrisi)")
md_lines.append("  ├── correlation_jail_top20.png (Jail korelasyonları)")
md_lines.append("  ├── correlation_probation_top20.png (Probation korelasyonları)")
md_lines.append("  ├── correlation_release_top20.png (Release korelasyonları)")
md_lines.append("  └── correlation_important_features.png (Önemli özellikler)")
md_lines.append("```\n")

# Yorumlar
md_lines.append("#### 💡 Önemli Bulgular ve Yorumlar\n")
md_lines.append("**Jail (Hapis Süresi) için:**")
md_lines.append("- Pozitif korelasyonlar → Bu özellikler artınca ceza süresi artar")
md_lines.append("- Negatif korelasyonlar → Bu özellikler artınca ceza süresi azalır")
md_lines.append("- Önceki suç geçmişi (prior_felony) genellikle yüksek korelasyonludur\n")

md_lines.append("**Model İçin Öneriler:**")
md_lines.append("1. 🔧 Yüksek korelasyonlu özellikleri (|r|>0.9) birleştir veya çıkar")
md_lines.append("2. 🔧 Hedef değişkenle zayıf korelasyonlu (|r|<0.05) özellikleri çıkarmayı düşün")
md_lines.append("3. 🔧 Feature selection için correlation threshold uygula")
md_lines.append("4. 🔧 XGBoost eğitiminde feature_importance değerlerini kontrol et\n")

md_lines.append("---\n")

# Dosyaya ekle
with open(SONUCLAR_PATH, 'a', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"✅ SONUCLAR.md güncellendi: {SONUCLAR_PATH}")
print("\n" + "=" * 70)
print("✅ ADIM 5.4 TAMAMLANDI!")
print("=" * 70)
print(f"📊 Toplam 5 grafik oluşturuldu.")
print(f"📁 Konum: {OUTPUT_DIR}")
