#!/usr/bin/env python3
"""
4 KATEGORİLİ MODEL OPTİMİZASYONU

Kategoriler:
  - Çok Hafif: 1-20 gün (ilk kez suç, küçük kabahatler)
  - Hafif: 21-60 gün (hafif suçlar)
  - Orta: 61-365 gün (orta vadeli cezalar)
  - Ağır: 366+ gün (uzun vadeli cezalar)

Amaç: Daha dengeli dağılım ile model performansını maksimize etmek
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("4 KATEGORİLİ MODEL OPTİMİZASYONU")
print("=" * 80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Çıktı klasörü
output_dir = Path('outputs/4_categories')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================
print("📂 Processed veri yükleniyor...")
df = pd.read_csv('wcld_Processed_For_Model.csv')
print(f"   Toplam kayıt: {len(df):,}\n")

# ============================================================================
# 2. 4 KATEGORİ OLUŞTUR
# ============================================================================
print("🔄 4 kategori oluşturuluyor...")

def categorize_4(jail):
    if jail <= 20:
        return 'CokHafif'
    elif jail <= 60:
        return 'Hafif'
    elif jail <= 365:
        return 'Orta'
    else:
        return 'Agir'

# Sadece jail > 0 olanları al
df_model = df[df['jail'] > 0].copy()
df_model['jail_category_4'] = df_model['jail'].apply(categorize_4)

# Dağılımı göster
print("\n📊 4 Kategori Dağılımı:")
dist = df_model['jail_category_4'].value_counts()
total = len(df_model)
for cat in ['CokHafif', 'Hafif', 'Orta', 'Agir']:
    if cat in dist.index:
        count = dist[cat]
        pct = count / total * 100
        print(f"   {cat:12s}: {count:7,} ({pct:5.2f}%)")

# ============================================================================
# 3. KARŞILAŞTIRMA: 3 vs 4 Kategori
# ============================================================================
print("\n📊 KARŞILAŞTIRMA: 3 Kategori vs 4 Kategori")
print("-" * 80)

# 3 kategori dağılımı (önceki)
with open('outputs/new_categories/new_category_model.pkl', 'rb') as f:
    model_3cat = pickle.load(f)

print("\n3 Kategori (1-60, 61-365, 366+):")
print("   Hafif: ~69%")
print("   Orta: ~26%")
print("   Ağır: ~5%")

print("\n4 Kategori (1-20, 21-60, 61-365, 366+):")
for cat in ['CokHafif', 'Hafif', 'Orta', 'Agir']:
    if cat in dist.index:
        pct = dist[cat] / total * 100
        print(f"   {cat:12s}: ~{pct:.0f}%")

# ============================================================================
# 4. TRAIN-TEST SPLIT (Stratified by 4 Categories)
# ============================================================================
print("\n🔀 Train-Test split (stratified by 4 kategoriler)...")

# Feature'lar ve target
feature_cols = [col for col in df_model.columns if col not in ['jail', 'release', 'probation', 'jail_category', 'jail_category_new', 'jail_category_4']]
X = df_model[feature_cols]
y = df_model['jail']
categories = df_model['jail_category_4']

# Stratified split
X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
    X, y, categories, 
    test_size=0.2, 
    random_state=42,
    stratify=categories
)

print(f"   Train: {len(X_train):,} kayıt")
print(f"   Test:  {len(X_test):,} kayıt")

# Test kategori dağılımı
print("\n📊 Test Set Kategori Dağılımı:")
test_dist = cat_test.value_counts()
for cat in ['CokHafif', 'Hafif', 'Orta', 'Agir']:
    if cat in test_dist.index:
        count = test_dist[cat]
        pct = count / len(cat_test) * 100
        print(f"   {cat:12s}: {count:7,} ({pct:5.2f}%)")

# ============================================================================
# 5. MODEL EĞİTİMİ (4 Kategori)
# ============================================================================
print("\n🤖 4 kategorili model eğitiliyor...")

model_4cat = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

model_4cat.fit(X_train, y_train)
print("   ✅ Eğitim tamamlandı!")

# ============================================================================
# 6. PERFORMANS KARŞILAŞTIRMASI
# ============================================================================
print("\n📊 Performans değerlendirmesi...")

y_pred_4cat = model_4cat.predict(X_test)

rmse_4 = np.sqrt(mean_squared_error(y_test, y_pred_4cat))
mae_4 = mean_absolute_error(y_test, y_pred_4cat)
r2_4 = r2_score(y_test, y_pred_4cat)

print(f"   RMSE: {rmse_4:.2f} gün")
print(f"   MAE:  {mae_4:.2f} gün")
print(f"   R²:   {r2_4:.4f}")

# 3 kategori performansı (önceki)
perf_3cat = model_3cat['new_performance']
rmse_3 = perf_3cat['rmse']
mae_3 = perf_3cat['mae']
r2_3 = perf_3cat['r2']

print("\n" + "=" * 80)
print("KARŞILAŞTIRMA: 3 Kategori vs 4 Kategori")
print("=" * 80)
print(f"{'Metrik':<20} {'3 Kategori':>20} {'4 Kategori':>20} {'İyileşme':>15}")
print("-" * 80)
print(f"{'RMSE (gün)':<20} {rmse_3:>20.2f} {rmse_4:>20.2f} {rmse_3 - rmse_4:>+15.2f}")
print(f"{'MAE (gün)':<20} {mae_3:>20.2f} {mae_4:>20.2f} {mae_3 - mae_4:>+15.2f}")
print(f"{'R² Score':<20} {r2_3:>20.4f} {r2_4:>20.4f} {r2_4 - r2_3:>+15.4f}")
print("=" * 80)

# İyileşme yüzdeleri
rmse_imp = (rmse_3 - rmse_4) / rmse_3 * 100
mae_imp = (mae_3 - mae_4) / mae_3 * 100
r2_imp = (r2_4 - r2_3) / abs(r2_3) * 100

print(f"\n💡 İyileşme Yüzdeleri:")
print(f"   RMSE: {rmse_imp:+.1f}%")
print(f"   MAE:  {mae_imp:+.1f}%")
print(f"   R²:   {r2_imp:+.1f}%")

# ============================================================================
# 7. KATEGORİ BAZLI DETAY
# ============================================================================
print("\n📊 Kategori bazlı performans (4 kategori)...")

results = []
for cat in ['CokHafif', 'Hafif', 'Orta', 'Agir']:
    mask = cat_test == cat
    if mask.sum() == 0:
        continue
    
    y_true_cat = y_test[mask]
    y_pred_cat = y_pred_4cat[mask]
    
    rmse_cat = np.sqrt(mean_squared_error(y_true_cat, y_pred_cat))
    mae_cat = mean_absolute_error(y_true_cat, y_pred_cat)
    r2_cat = r2_score(y_true_cat, y_pred_cat)
    
    results.append({
        'Kategori': cat,
        'N': mask.sum(),
        'RMSE': rmse_cat,
        'MAE': mae_cat,
        'R²': r2_cat,
        'Ort_Gerçek': y_true_cat.mean(),
        'Ort_Tahmin': y_pred_cat.mean()
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# CSV kaydet
df_results.to_csv(output_dir / '4_kategori_performans.csv', index=False)

# ============================================================================
# 8. GÖRSELLEŞTİRME
# ============================================================================
print("\n📊 Grafikler oluşturuluyor...")

# Grafik 1: Kategori dağılımı karşılaştırma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 3 Kategori
cats_3 = ['Hafif\n(1-60)', 'Orta\n(61-365)', 'Ağır\n(366+)']
vals_3 = [69, 26, 5]
ax1.bar(cats_3, vals_3, color=['lightgreen', 'orange', 'red'], alpha=0.7)
ax1.set_ylabel('Veri Yüzdesi (%)', fontsize=11)
ax1.set_title('3 Kategori Dağılımı', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(vals_3):
    ax1.text(i, v + 1, f'{v}%', ha='center', fontweight='bold')

# 4 Kategori
cats_4 = ['Çok Hafif\n(1-20)', 'Hafif\n(21-60)', 'Orta\n(61-365)', 'Ağır\n(366+)']
vals_4 = [dist['CokHafif']/total*100, dist['Hafif']/total*100, dist['Orta']/total*100, dist['Agir']/total*100]
ax2.bar(cats_4, vals_4, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
ax2.set_ylabel('Veri Yüzdesi (%)', fontsize=11)
ax2.set_title('4 Kategori Dağılımı (ÖNERİLEN)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(vals_4):
    ax2.text(i, v + 1, f'{v:.0f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'kategori_dagilim_karsilastirma.png', dpi=300, bbox_inches='tight')
print(f"   ✅ {output_dir / 'kategori_dagilim_karsilastirma.png'}")
plt.close()

# Grafik 2: MAE karşılaştırma
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_results))
bars = ax.bar(x, df_results['MAE'], color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
ax.set_xlabel('Kategori', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE (gün)', fontsize=12, fontweight='bold')
ax.set_title('4 Kategori - Kategori Bazlı MAE Performansı', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_results['Kategori'])
ax.grid(axis='y', alpha=0.3)

for bar, mae in zip(bars, df_results['MAE']):
    ax.text(bar.get_x() + bar.get_width()/2., mae,
            f'{mae:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '4_kategori_mae_performance.png', dpi=300, bbox_inches='tight')
print(f"   ✅ {output_dir / '4_kategori_mae_performance.png'}")
plt.close()

# ============================================================================
# 9. MODELİ KAYDET
# ============================================================================
print("\n💾 Model kaydediliyor...")

model_info = {
    'model': model_4cat,
    'categories': 'CokHafif: 1-20, Hafif: 21-60, Orta: 61-365, Agir: 366+',
    'performance_3cat': {'rmse': rmse_3, 'mae': mae_3, 'r2': r2_3},
    'performance_4cat': {'rmse': rmse_4, 'mae': mae_4, 'r2': r2_4},
    'improvements': {'rmse_pct': rmse_imp, 'mae_pct': mae_imp, 'r2_pct': r2_imp},
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(output_dir / '4_category_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

# Yeni train/test verileri kaydet
new_model_data = Path('model_data_4_categories')
new_model_data.mkdir(exist_ok=True)

X_train.to_csv(new_model_data / 'X_train.csv', index=False)
X_test.to_csv(new_model_data / 'X_test.csv', index=False)
pd.DataFrame({'jail': y_train, 'jail_category': cat_train}).to_csv(new_model_data / 'y_train.csv', index=False)
pd.DataFrame({'jail': y_test, 'jail_category': cat_test}).to_csv(new_model_data / 'y_test.csv', index=False)

print(f"   ✅ Model: {output_dir / '4_category_model.pkl'}")
print(f"   ✅ Veriler: {new_model_data}/")

# ============================================================================
# 10. SONUÇ ÖZETİ
# ============================================================================
print("\n" + "=" * 80)
print("SONUÇ ÖZETİ")
print("=" * 80)

if rmse_imp > 0:
    print(f"✅ 4 KATEGORİ DAHA İYİ! RMSE {rmse_imp:.1f}% iyileşti.")
    print(f"✅ KARAR: 4 kategoriyi kullan!")
else:
    print(f"⚠️  3 kategori daha iyi. RMSE {rmse_imp:.1f}% değişim.")
    print(f"💡 KARAR: 3 kategoriyi kullanmaya devam et.")

if r2_4 > r2_3:
    print(f"✅ R² Score arttı: {r2_3:.4f} → {r2_4:.4f} (+{r2_imp:.1f}%)")
else:
    print(f"⚠️  R² Score düştü: {r2_3:.4f} → {r2_4:.4f} ({r2_imp:.1f}%)")

print("\n📌 4 Kategori Performansı:")
for idx, row in df_results.iterrows():
    print(f"   {row['Kategori']:12s}: N={row['N']:6,}, MAE={row['MAE']:6.1f} gün, R²={row['R²']:7.4f}")

print("\n💡 SONRAKİ ADIM:")
if rmse_imp > 1 or r2_4 > r2_3:
    print("   ✅ 4 kategori başarılı! Kategori bazlı ayrı modeller dene.")
else:
    print("   ✅ 3 kategori yeterli. Demographic parity analizine geç.")

print("\n" + "=" * 80)
print(f"✅ ANALİZ TAMAMLANDI! Çıktılar: {output_dir}/")
print("=" * 80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
