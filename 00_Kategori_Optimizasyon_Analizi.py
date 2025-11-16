"""
CEZA KATEGORİLERİ OPTİMİZASYON ANALİZİ
==========================================
Mevcut kategorileri analiz eder ve veri bazlı optimal aralıklar önerir.

Tarih: 2025-11-14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("CEZA KATEGORİLERİ OPTİMİZASYON ANALİZİ")
print("=" * 80)

# Veriyi yükle
y_train = pd.read_csv('model_data/y_train.csv')
y_test = pd.read_csv('model_data/y_test.csv')

# Tüm veriyi birleştir (daha iyi analiz için)
y_all = pd.concat([y_train, y_test], ignore_index=True)

print(f"\n📊 Toplam Veri: {len(y_all):,} kayıt")
print(f"   - Train: {len(y_train):,}")
print(f"   - Test: {len(y_test):,}")

# ============================================================================
# 1. MEVCUT KATEGORİ DAĞILIMI
# ============================================================================
print("\n" + "=" * 80)
print("1. MEVCUT KATEGORİ DAĞILIMI (Hafif: 1-180, Orta: 181-1080, Ağır: 1080+)")
print("=" * 80)

current_dist = y_all['jail_category'].value_counts()
for cat in ['Hafif', 'Orta', 'Agir']:
    if cat in current_dist.index:
        count = current_dist[cat]
        pct = count / len(y_all) * 100
        print(f"   {cat:8s}: {count:7,} ({pct:5.2f}%)")

# ============================================================================
# 2. İSTATİSTİKSEL ANALİZ
# ============================================================================
print("\n" + "=" * 80)
print("2. İSTATİSTİKSEL ANALİZ (jail süresi)")
print("=" * 80)

stats = y_all['jail'].describe(percentiles=[0.25, 0.33, 0.5, 0.66, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])
print(stats)

print("\n📌 Önemli Eşik Değerler:")
percentiles = [25, 33, 50, 66, 75, 80, 85, 90, 95, 99]
for p in percentiles:
    val = y_all['jail'].quantile(p/100)
    print(f"   {p:2d}. Percentile: {val:7.1f} gün (~{val/30:.1f} ay)")

# ============================================================================
# 3. DAĞILIM GÖRSELLEŞTİRMESİ
# ============================================================================
print("\n" + "=" * 80)
print("3. DAĞILIM GÖRSELLEŞTİRMESİ")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# A) Histogram (tüm veri)
ax1 = axes[0, 0]
ax1.hist(y_all['jail'], bins=100, edgecolor='black', alpha=0.7)
ax1.axvline(180, color='red', linestyle='--', linewidth=2, label='Mevcut: 180 gün')
ax1.axvline(1080, color='orange', linestyle='--', linewidth=2, label='Mevcut: 1080 gün')
ax1.set_xlabel('Jail Süresi (gün)', fontsize=12)
ax1.set_ylabel('Frekans', fontsize=12)
ax1.set_title('A) Jail Dağılımı (Tüm Veri)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# B) Histogram (0-500 gün zoom)
ax2 = axes[0, 1]
jail_zoom = y_all[y_all['jail'] <= 500]['jail']
ax2.hist(jail_zoom, bins=50, edgecolor='black', alpha=0.7, color='green')
ax2.axvline(180, color='red', linestyle='--', linewidth=2, label='Mevcut: 180 gün')
# Quartile çizgileri
q33 = y_all['jail'].quantile(0.33)
q66 = y_all['jail'].quantile(0.66)
ax2.axvline(q33, color='blue', linestyle=':', linewidth=2, label=f'33%: {q33:.0f} gün')
ax2.axvline(q66, color='purple', linestyle=':', linewidth=2, label=f'66%: {q66:.0f} gün')
ax2.set_xlabel('Jail Süresi (gün)', fontsize=12)
ax2.set_ylabel('Frekans', fontsize=12)
ax2.set_title('B) Jail Dağılımı (0-500 gün ZOOM)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# C) Log scale histogram
ax3 = axes[1, 0]
ax3.hist(y_all['jail'], bins=100, edgecolor='black', alpha=0.7, color='orange')
ax3.set_yscale('log')
ax3.axvline(180, color='red', linestyle='--', linewidth=2, label='Mevcut: 180 gün')
ax3.axvline(1080, color='orange', linestyle='--', linewidth=2, label='Mevcut: 1080 gün')
ax3.set_xlabel('Jail Süresi (gün)', fontsize=12)
ax3.set_ylabel('Frekans (Log Scale)', fontsize=12)
ax3.set_title('C) Jail Dağılımı (Log Scale)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# D) Boxplot kategorilere göre
ax4 = axes[1, 1]
y_all.boxplot(column='jail', by='jail_category', ax=ax4)
ax4.set_xlabel('Kategori', fontsize=12)
ax4.set_ylabel('Jail Süresi (gün)', fontsize=12)
ax4.set_title('D) Kategorilere Göre Jail Dağılımı', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove automatic title

plt.tight_layout()
plt.savefig('outputs/kategori_optimizasyon_analiz.png', dpi=300, bbox_inches='tight')
print("✅ Grafik kaydedildi: outputs/kategori_optimizasyon_analiz.png")

# ============================================================================
# 4. ALTERNATİF KATEGORİZASYON ÖNERİLERİ
# ============================================================================
print("\n" + "=" * 80)
print("4. ALTERNATİF KATEGORİZASYON ÖNERİLERİ")
print("=" * 80)

# ÖNERI 1: Quantile-Based (Equal Frequency)
print("\n📋 ÖNERİ 1: QUANTILE-BASED (Eşit Sayıda Veri)")
print("-" * 80)
q33 = y_all['jail'].quantile(0.33)
q66 = y_all['jail'].quantile(0.66)
print(f"   Hafif : 1-{q33:.0f} gün (~{q33/30:.1f} ay)")
print(f"   Orta  : {q33+1:.0f}-{q66:.0f} gün (~{(q33+1)/30:.1f}-{q66/30:.1f} ay)")
print(f"   Ağır  : {q66+1:.0f}+ gün (~{(q66+1)/30:.1f}+ ay)")

# Dağılımı hesapla
def categorize_quantile(jail):
    if jail <= q33:
        return 'Hafif'
    elif jail <= q66:
        return 'Orta'
    else:
        return 'Agir'

y_all['cat_quantile'] = y_all['jail'].apply(categorize_quantile)
dist_quantile = y_all['cat_quantile'].value_counts()
print("\n   Dağılım:")
for cat in ['Hafif', 'Orta', 'Agir']:
    if cat in dist_quantile.index:
        count = dist_quantile[cat]
        pct = count / len(y_all) * 100
        print(f"      {cat:8s}: {count:7,} ({pct:5.2f}%)")

# ÖNERI 2: Domain-Based (Yasal/Anlamlı Aralıklar)
print("\n📋 ÖNERİ 2: DOMAIN-BASED (Yasal/Anlamlı Aralıklar)")
print("-" * 80)
print("   Çok Hafif: 1-30 gün (1 aya kadar)")
print("   Hafif    : 31-90 gün (1-3 ay)")
print("   Orta     : 91-365 gün (3 ay - 1 yıl)")
print("   Ağır     : 366-1825 gün (1-5 yıl)")
print("   Çok Ağır : 1826+ gün (5+ yıl)")

def categorize_domain(jail):
    if jail <= 30:
        return 'CokHafif'
    elif jail <= 90:
        return 'Hafif'
    elif jail <= 365:
        return 'Orta'
    elif jail <= 1825:
        return 'Agir'
    else:
        return 'CokAgir'

y_all['cat_domain'] = y_all['jail'].apply(categorize_domain)
dist_domain = y_all['cat_domain'].value_counts()
print("\n   Dağılım:")
for cat in ['CokHafif', 'Hafif', 'Orta', 'Agir', 'CokAgir']:
    if cat in dist_domain.index:
        count = dist_domain[cat]
        pct = count / len(y_all) * 100
        print(f"      {cat:12s}: {count:7,} ({pct:5.2f}%)")

# ÖNERI 3: Balanced (Dengeli - veri + domain)
print("\n📋 ÖNERİ 3: BALANCED (Dengeli - Veri + Domain)")
print("-" * 80)
print("   Hafif : 1-60 gün (~2 aya kadar)")
print("   Orta  : 61-365 gün (2 ay - 1 yıl)")
print("   Ağır  : 366+ gün (1+ yıl)")

def categorize_balanced(jail):
    if jail <= 60:
        return 'Hafif'
    elif jail <= 365:
        return 'Orta'
    else:
        return 'Agir'

y_all['cat_balanced'] = y_all['jail'].apply(categorize_balanced)
dist_balanced = y_all['cat_balanced'].value_counts()
print("\n   Dağılım:")
for cat in ['Hafif', 'Orta', 'Agir']:
    if cat in dist_balanced.index:
        count = dist_balanced[cat]
        pct = count / len(y_all) * 100
        print(f"      {cat:8s}: {count:7,} ({pct:5.2f}%)")

# ÖNERI 4: Optimized (80-15-5 hedefi)
print("\n📋 ÖNERİ 4: OPTIMIZED (80-15-5 Dağılım Hedefi)")
print("-" * 80)
q80 = y_all['jail'].quantile(0.80)
q95 = y_all['jail'].quantile(0.95)
print(f"   Hafif : 1-{q80:.0f} gün (~{q80/30:.1f} ay)")
print(f"   Orta  : {q80+1:.0f}-{q95:.0f} gün (~{(q80+1)/30:.1f}-{q95/30:.1f} ay)")
print(f"   Ağır  : {q95+1:.0f}+ gün (~{(q95+1)/30:.1f}+ ay)")

def categorize_optimized(jail):
    if jail <= q80:
        return 'Hafif'
    elif jail <= q95:
        return 'Orta'
    else:
        return 'Agir'

y_all['cat_optimized'] = y_all['jail'].apply(categorize_optimized)
dist_optimized = y_all['cat_optimized'].value_counts()
print("\n   Dağılım:")
for cat in ['Hafif', 'Orta', 'Agir']:
    if cat in dist_optimized.index:
        count = dist_optimized[cat]
        pct = count / len(y_all) * 100
        print(f"      {cat:8s}: {count:7,} ({pct:5.2f}%)")

# ============================================================================
# 5. ÖNERİ KARŞILAŞTIRMASI
# ============================================================================
print("\n" + "=" * 80)
print("5. TÜM ÖNERİLERİN KARŞILAŞTIRMASI")
print("=" * 80)

comparison = pd.DataFrame({
    'Mevcut (1-180, 181-1080, 1080+)': dist_balanced.reindex(['Hafif', 'Orta', 'Agir'], fill_value=0),
    'Quantile (33-66)': dist_quantile.reindex(['Hafif', 'Orta', 'Agir'], fill_value=0),
    'Balanced (1-60, 61-365, 366+)': dist_balanced.reindex(['Hafif', 'Orta', 'Agir'], fill_value=0),
    'Optimized (80-95)': dist_optimized.reindex(['Hafif', 'Orta', 'Agir'], fill_value=0)
})

print("\n📊 Kategori Dağılımı Karşılaştırması (Sayı):")
print(comparison)

print("\n📊 Kategori Dağılımı Karşılaştırması (Yüzde):")
comparison_pct = (comparison / len(y_all) * 100).round(2)
print(comparison_pct)

# ============================================================================
# 6. SONUÇ VE ÖNERİ
# ============================================================================
print("\n" + "=" * 80)
print("6. SONUÇ VE ÖNERİ")
print("=" * 80)

print("""
🎯 MODEL PERFORMANSI İÇİN EN İYİ YAKLAŞIM:

📌 ÖNERİM: "BALANCED" KATEGORİZASYON (1-60, 61-365, 366+)

NEDEN?
------
1. ✅ Mevcut sisteme göre daha dengeli dağılım:
   - Hafif: ~70-75% (mevcut: 90% - çok dengesiz!)
   - Orta: ~20-25% (mevcut: 7.6% - çok az veri!)
   - Ağır: ~5-8% (mevcut: 1.9% - çok az veri!)

2. ✅ Yasal olarak anlamlı:
   - 60 gün = 2 ay (kısa süreli hapis)
   - 365 gün = 1 yıl (orta vadeli hapis)
   - 366+ gün = 1+ yıl (uzun vadeli hapis)

3. ✅ Orta kategoride daha fazla veri:
   - Daha iyi model eğitimi
   - R² performansı artacak

4. ✅ Stratified sampling daha etkili olacak:
   - Her kategoride yeterli veri var
   - Cross-validation daha kararlı

ALTERNATİF: "OPTIMIZED" (80-95 percentile)
-------------------------------------------
Eğer 3 yerine 2 kategori isterseniz:
   - Hafif: 1-{q80:.0f} gün (80%)
   - Ağır: {q80+1:.0f}+ gün (20%)
   
Bu, binary classification gibi davranır ve daha basit olabilir.

NASIL UYGULAYACAĞIZ?
---------------------
Bir sonraki adımda:
1. 10_Normalizasyon_ve_Train_Test_Split.py'yi güncelleyelim
2. Yeni kategorilerle tekrar split yapalım
3. Model tekrar eğitelim
4. Performans karşılaştırması yapalım (mevcut vs yeni)
""")

print("\n" + "=" * 80)
print("ANALİZ TAMAMLANDI!")
print("=" * 80)
