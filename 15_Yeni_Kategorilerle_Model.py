#!/usr/bin/env python3
"""
YENİ KATEGORİLERLE VERİ SPLIT VE MODEL EĞİTİMİ

Kategori optimizasyon analizinden çıkan ÖNERİ:
  - Hafif: 1-60 gün (69.16% veri)
  - Orta: 61-365 gün (26.11% veri)
  - Ağır: 366+ gün (4.74% veri)

Bu, mevcut kategorilere göre ÇOK DAHA DENGELİ!
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("YENİ KATEGORİLERLE MODEL EĞİTİMİ")
print("=" * 80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Çıktı klasörü
output_dir = Path('outputs/new_categories')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================
print("📂 Processed veri yükleniyor...")
df = pd.read_csv('wcld_Processed_For_Model.csv')
print(f"   Toplam kayıt: {len(df):,}\n")

# ============================================================================
# 2. YENİ KATEGORİLER OLUŞTUR
# ============================================================================
print("🔄 Yeni kategoriler oluşturuluyor (BALANCED: 1-60, 61-365, 366+)...")

def categorize_new(jail):
    if jail <= 60:
        return 'Hafif'
    elif jail <= 365:
        return 'Orta'
    else:
        return 'Agir'

# Sadece jail > 0 olanları al
df_model = df[df['jail'] > 0].copy()
df_model['jail_category_new'] = df_model['jail'].apply(categorize_new)

# Dağılımı göster
print("\n📊 YENİ Kategori Dağılımı:")
dist = df_model['jail_category_new'].value_counts()
for cat in ['Hafif', 'Orta', 'Agir']:
    if cat in dist.index:
        count = dist[cat]
        pct = count / len(df_model) * 100
        print(f"   {cat:8s}: {count:7,} ({pct:5.2f}%)")

# Eski kategorileri göster (karşılaştırma için)
print("\n📊 ESKİ Kategori Dağılımı (Karşılaştırma):")
if 'jail_category' in df_model.columns:
    dist_old = df_model['jail_category'].value_counts()
    for cat in ['Hafif', 'Orta', 'Agir']:
        if cat in dist_old.index:
            count = dist_old[cat]
            pct = count / len(df_model) * 100
            print(f"   {cat:8s}: {count:7,} ({pct:5.2f}%)")

# ============================================================================
# 3. TRAIN-TEST SPLIT (Stratified by New Category)
# ============================================================================
print("\n🔀 Train-Test split (stratified by yeni kategoriler)...")

# Feature'lar ve target
feature_cols = [col for col in df_model.columns if col not in ['jail', 'release', 'probation', 'jail_category', 'jail_category_new']]
X = df_model[feature_cols]
y = df_model['jail']
categories = df_model['jail_category_new']

# Stratified split
X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
    X, y, categories, 
    test_size=0.2, 
    random_state=42,
    stratify=categories
)

print(f"   Train: {len(X_train):,} kayıt")
print(f"   Test:  {len(X_test):,} kayıt")

# Train/Test kategori dağılımlarını kontrol et
print("\n📊 Train Set Kategori Dağılımı:")
train_dist = cat_train.value_counts()
for cat in ['Hafif', 'Orta', 'Agir']:
    if cat in train_dist.index:
        count = train_dist[cat]
        pct = count / len(cat_train) * 100
        print(f"   {cat:8s}: {count:7,} ({pct:5.2f}%)")

print("\n📊 Test Set Kategori Dağılımı:")
test_dist = cat_test.value_counts()
for cat in ['Hafif', 'Orta', 'Agir']:
    if cat in test_dist.index:
        count = test_dist[cat]
        pct = count / len(cat_test) * 100
        print(f"   {cat:8s}: {count:7,} ({pct:5.2f}%)")

# ============================================================================
# 4. ESKİ MODEL PERFORMANSI (Baseline)
# ============================================================================
print("\n📊 ESKİ model performansı (mevcut kategorilerle)...")
with open('outputs/model/xgboost_jail_model.pkl', 'rb') as f:
    old_model = pickle.load(f)

# Eski test seti yükle
X_test_old = pd.read_csv('model_data/X_test.csv')
y_test_old = pd.read_csv('model_data/y_test.csv')

y_pred_old = old_model.predict(X_test_old)
old_rmse = np.sqrt(mean_squared_error(y_test_old['jail'], y_pred_old))
old_mae = mean_absolute_error(y_test_old['jail'], y_pred_old)
old_r2 = r2_score(y_test_old['jail'], y_pred_old)

print(f"   RMSE: {old_rmse:.2f} gün")
print(f"   MAE:  {old_mae:.2f} gün")
print(f"   R²:   {old_r2:.4f}")

# ============================================================================
# 5. YENİ MODEL EĞİTİMİ (Aynı Hyperparameters)
# ============================================================================
print("\n🤖 YENİ kategorilerle model eğitiliyor...")
print("   (Aynı hyperparameter'lar kullanılıyor: n_estimators=300, max_depth=3, lr=0.05)")

new_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

new_model.fit(X_train, y_train)
print("   ✅ Eğitim tamamlandı!")

# ============================================================================
# 6. YENİ MODEL PERFORMANSI
# ============================================================================
print("\n📊 YENİ model performansı...")
y_pred_new = new_model.predict(X_test)

new_rmse = np.sqrt(mean_squared_error(y_test, y_pred_new))
new_mae = mean_absolute_error(y_test, y_pred_new)
new_r2 = r2_score(y_test, y_pred_new)

print(f"   RMSE: {new_rmse:.2f} gün")
print(f"   MAE:  {new_mae:.2f} gün")
print(f"   R²:   {new_r2:.4f}")

# ============================================================================
# 7. KARŞILAŞTIRMA
# ============================================================================
print("\n" + "=" * 80)
print("KARŞILAŞTIRMA: ESKİ vs YENİ Kategoriler")
print("=" * 80)
print(f"{'Metrik':<20} {'ESKİ (1-180,181-1080,1080+)':>30} {'YENİ (1-60,61-365,366+)':>30} {'İyileşme':>15}")
print("-" * 80)
print(f"{'RMSE (gün)':<20} {old_rmse:>30.2f} {new_rmse:>30.2f} {old_rmse - new_rmse:>+15.2f}")
print(f"{'MAE (gün)':<20} {old_mae:>30.2f} {new_mae:>30.2f} {old_mae - new_mae:>+15.2f}")
print(f"{'R² Score':<20} {old_r2:>30.4f} {new_r2:>30.4f} {new_r2 - old_r2:>+15.4f}")
print("=" * 80)

# İyileşme yüzdeleri
rmse_imp = (old_rmse - new_rmse) / old_rmse * 100
mae_imp = (old_mae - new_mae) / old_mae * 100
r2_imp = (new_r2 - old_r2) / abs(old_r2) * 100

print(f"\n💡 İyileşme Yüzdeleri:")
print(f"   RMSE: {rmse_imp:+.1f}%")
print(f"   MAE:  {mae_imp:+.1f}%")
print(f"   R²:   {r2_imp:+.1f}%")

# ============================================================================
# 8. KATEGORİ BAZLI PERFORMANS (YENİ MODEL)
# ============================================================================
print("\n📊 Kategori bazlı performans (YENİ kategorilerle)...")

results = []
for cat in ['Hafif', 'Orta', 'Agir']:
    mask = cat_test == cat
    if mask.sum() == 0:
        continue
    
    y_true_cat = y_test[mask]
    y_pred_cat = y_pred_new[mask]
    
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
df_results.to_csv(output_dir / 'yeni_kategori_performans.csv', index=False)

# ============================================================================
# 9. MODELİ VE VERİYİ KAYDET
# ============================================================================
print("\n💾 Yeni model ve veriler kaydediliyor...")

# Model kaydet
model_info = {
    'model': new_model,
    'categories': 'Hafif: 1-60, Orta: 61-365, Agir: 366+',
    'old_performance': {'rmse': old_rmse, 'mae': old_mae, 'r2': old_r2},
    'new_performance': {'rmse': new_rmse, 'mae': new_mae, 'r2': new_r2},
    'improvements': {'rmse_pct': rmse_imp, 'mae_pct': mae_imp, 'r2_pct': r2_imp},
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(output_dir / 'new_category_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

# Yeni train/test verileri kaydet
new_model_data = Path('model_data_new_categories')
new_model_data.mkdir(exist_ok=True)

X_train.to_csv(new_model_data / 'X_train.csv', index=False)
X_test.to_csv(new_model_data / 'X_test.csv', index=False)
pd.DataFrame({'jail': y_train, 'jail_category': cat_train}).to_csv(new_model_data / 'y_train.csv', index=False)
pd.DataFrame({'jail': y_test, 'jail_category': cat_test}).to_csv(new_model_data / 'y_test.csv', index=False)

print(f"   ✅ Model: {output_dir / 'new_category_model.pkl'}")
print(f"   ✅ Veriler: {new_model_data}/")

# ============================================================================
# 10. SONUÇ ÖZETİ
# ============================================================================
print("\n" + "=" * 80)
print("SONUÇ ÖZETİ")
print("=" * 80)

if rmse_imp > 0:
    print(f"✅ YENİ kategoriler BAŞARILI! RMSE {rmse_imp:.1f}% iyileşti.")
else:
    print(f"⚠️  YENİ kategoriler beklenen etkiyi yaratmadı. RMSE {rmse_imp:.1f}% değişim.")

if new_r2 > old_r2:
    print(f"✅ R² Score arttı: {old_r2:.4f} → {new_r2:.4f} (+{r2_imp:.1f}%)")
else:
    print(f"⚠️  R² Score düştü: {old_r2:.4f} → {new_r2:.4f} ({r2_imp:.1f}%)")

print("\n📌 Kategori Bazlı Performans:")
for idx, row in df_results.iterrows():
    print(f"   {row['Kategori']:8s}: N={row['N']:6,}, MAE={row['MAE']:6.1f} gün, R²={row['R²']:7.4f}")

print("\n💡 SONRAKİ ADIMLAR:")
if rmse_imp > 2 or new_r2 > old_r2:
    print("   1. ✅ Yeni kategorileri kullan!")
    print("   2. 🔧 Kategori bazlı ayrı modeller dene (Hafif/Orta/Ağır için)")
    print("   3. 🔧 Hyperparameter tuning ile daha da iyileştir")
else:
    print("   1. ⚠️  Kategori değişikliği küçük etki yarattı")
    print("   2. 🔧 Kategori bazlı ayrı modeller dene")
    print("   3. 🔧 Ensemble methods dene")

print("\n" + "=" * 80)
print(f"✅ ANALİZ TAMAMLANDI! Tüm çıktılar: {output_dir}/")
print("=" * 80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
