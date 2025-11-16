#!/usr/bin/env python3
"""
ADIM 1: LOG TRANSFORMATION İLE MODEL İYİLEŞTİRME

Log transformation, sağa çarpık (skewed) dağılımları normalize eder.
Ceza süreleri gibi uzun kuyruklu verilerde çok etkilidir.

Beklenen İyileştirme:
  - RMSE: 577 → 400-450 gün
  - Özellikle ağır cezalarda (1080+ gün) büyük iyileşme
  - R² artışı bekleniyor

Süre: ~2-3 saat (GridSearchCV dahil)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("LOG TRANSFORMATION İLE MODEL İYİLEŞTİRME")
print("=" * 80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Klasör oluştur
output_dir = Path('outputs/log_transformation')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================
print("📂 Veri yükleniyor...")
X_train = pd.read_csv('model_data/X_train.csv')
X_test = pd.read_csv('model_data/X_test.csv')
y_train = pd.read_csv('model_data/y_train.csv')
y_test = pd.read_csv('model_data/y_test.csv')

print(f"   Train: {len(X_train):,} kayıt")
print(f"   Test: {len(X_test):,} kayıt\n")

# ============================================================================
# 2. MEVCUT MODELİ YÜKLE (Baseline)
# ============================================================================
print("📊 Mevcut model performansı (Baseline)...")
with open('outputs/model/xgboost_jail_model.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

y_pred_baseline = baseline_model.predict(X_test)

baseline_rmse = np.sqrt(mean_squared_error(y_test['jail'], y_pred_baseline))
baseline_mae = mean_absolute_error(y_test['jail'], y_pred_baseline)
baseline_r2 = r2_score(y_test['jail'], y_pred_baseline)

print(f"   RMSE: {baseline_rmse:.2f} gün")
print(f"   MAE:  {baseline_mae:.2f} gün")
print(f"   R²:   {baseline_r2:.4f}\n")

# ============================================================================
# 3. LOG TRANSFORMATION UYGULA
# ============================================================================
print("🔄 Log transformation uygulanıyor...")

# log1p: log(1 + x) - sıfır değerleri için güvenli
y_train_log = np.log1p(y_train['jail'])
y_test_log = np.log1p(y_test['jail'])

print(f"   Orijinal ölçek - Min: {y_train['jail'].min():.1f}, Max: {y_train['jail'].max():.1f}")
print(f"   Log ölçek - Min: {y_train_log.min():.3f}, Max: {y_train_log.max():.3f}\n")

# ============================================================================
# 4. LOG-SCALE MODEL EĞİTİMİ (Baseline Hyperparameters)
# ============================================================================
print("🤖 Log-scale model eğitiliyor (baseline hyperparameters)...")

# Mevcut projedeki en iyi parametreleri kullan
log_model_baseline = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

log_model_baseline.fit(X_train, y_train_log)
print("   ✅ Eğitim tamamlandı!\n")

# Tahmin yap (log ölçekte)
y_pred_log = log_model_baseline.predict(X_test)

# Log ölçekten geri çevir: expm1(x) = exp(x) - 1
y_pred_original = np.expm1(y_pred_log)

# Negatif tahminleri sıfıra çek (güvenlik)
y_pred_original = np.maximum(y_pred_original, 0)

# ============================================================================
# 5. PERFORMANS KARŞILAŞTIRMASI
# ============================================================================
print("📊 Log-transformation model performansı...")

log_rmse = np.sqrt(mean_squared_error(y_test['jail'], y_pred_original))
log_mae = mean_absolute_error(y_test['jail'], y_pred_original)
log_r2 = r2_score(y_test['jail'], y_pred_original)

print(f"   RMSE: {log_rmse:.2f} gün")
print(f"   MAE:  {log_mae:.2f} gün")
print(f"   R²:   {log_r2:.4f}\n")

print("=" * 80)
print("KARŞILAŞTIRMA: Baseline vs Log-Transformation")
print("=" * 80)
print(f"{'Metrik':<20} {'Baseline':>15} {'Log-Transform':>15} {'İyileşme':>15}")
print("-" * 80)
print(f"{'RMSE (gün)':<20} {baseline_rmse:>15.2f} {log_rmse:>15.2f} {baseline_rmse - log_rmse:>+15.2f}")
print(f"{'MAE (gün)':<20} {baseline_mae:>15.2f} {log_mae:>15.2f} {baseline_mae - log_mae:>+15.2f}")
print(f"{'R² Score':<20} {baseline_r2:>15.4f} {log_r2:>15.4f} {log_r2 - baseline_r2:>+15.4f}")
print("=" * 80 + "\n")

# İyileşme yüzdesi
rmse_improvement = (baseline_rmse - log_rmse) / baseline_rmse * 100
mae_improvement = (baseline_mae - log_mae) / baseline_mae * 100
r2_improvement = (log_r2 - baseline_r2) / abs(baseline_r2) * 100

print(f"💡 İyileşme Yüzdeleri:")
print(f"   RMSE: {rmse_improvement:+.1f}%")
print(f"   MAE:  {mae_improvement:+.1f}%")
print(f"   R²:   {r2_improvement:+.1f}%\n")

# ============================================================================
# 6. KATEGORİ BAZLI PERFORMANS
# ============================================================================
print("📊 Kategori bazlı performans analizi...")

# Test kategorilerini al
test_categories = y_test['jail_category']

results = []
for cat in ['Hafif', 'Orta', 'Agir']:
    mask = test_categories == cat
    if mask.sum() == 0:
        continue
    
    y_true_cat = y_test['jail'][mask]
    y_pred_baseline_cat = y_pred_baseline[mask]
    y_pred_log_cat = y_pred_original[mask]
    
    # Baseline
    rmse_base = np.sqrt(mean_squared_error(y_true_cat, y_pred_baseline_cat))
    mae_base = mean_absolute_error(y_true_cat, y_pred_baseline_cat)
    r2_base = r2_score(y_true_cat, y_pred_baseline_cat)
    
    # Log
    rmse_log = np.sqrt(mean_squared_error(y_true_cat, y_pred_log_cat))
    mae_log = mean_absolute_error(y_true_cat, y_pred_log_cat)
    r2_log = r2_score(y_true_cat, y_pred_log_cat)
    
    results.append({
        'Kategori': cat,
        'N': mask.sum(),
        'Baseline_RMSE': rmse_base,
        'Log_RMSE': rmse_log,
        'RMSE_Fark': rmse_base - rmse_log,
        'Baseline_MAE': mae_base,
        'Log_MAE': mae_log,
        'MAE_Fark': mae_base - mae_log,
        'Baseline_R2': r2_base,
        'Log_R2': r2_log,
        'R2_Fark': r2_log - r2_base
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print()

# CSV kaydet
df_results.to_csv(output_dir / 'kategori_performans_karsilastirma.csv', index=False)

# ============================================================================
# 7. GÖRSELLEŞTİRME
# ============================================================================
print("📊 Grafikler oluşturuluyor...")

# Grafik 1: Prediction vs Actual (Baseline vs Log)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Baseline
ax1 = axes[0]
ax1.scatter(y_test['jail'], y_pred_baseline, alpha=0.3, s=10)
ax1.plot([0, y_test['jail'].max()], [0, y_test['jail'].max()], 'r--', linewidth=2)
ax1.set_xlabel('Gerçek Ceza (gün)', fontsize=12)
ax1.set_ylabel('Tahmin (gün)', fontsize=12)
ax1.set_title(f'Baseline Model\nRMSE: {baseline_rmse:.1f}, MAE: {baseline_mae:.1f}, R²: {baseline_r2:.4f}', 
              fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Log-Transform
ax2 = axes[1]
ax2.scatter(y_test['jail'], y_pred_original, alpha=0.3, s=10, color='green')
ax2.plot([0, y_test['jail'].max()], [0, y_test['jail'].max()], 'r--', linewidth=2)
ax2.set_xlabel('Gerçek Ceza (gün)', fontsize=12)
ax2.set_ylabel('Tahmin (gün)', fontsize=12)
ax2.set_title(f'Log-Transform Model\nRMSE: {log_rmse:.1f}, MAE: {log_mae:.1f}, R²: {log_r2:.4f}', 
              fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_vs_actual_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✅ {output_dir / 'prediction_vs_actual_comparison.png'}")
plt.close()

# Grafik 2: Kategori bazlı MAE karşılaştırma
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_results))
width = 0.35

bars1 = ax.bar(x - width/2, df_results['Baseline_MAE'], width, label='Baseline', color='skyblue')
bars2 = ax.bar(x + width/2, df_results['Log_MAE'], width, label='Log-Transform', color='lightgreen')

ax.set_xlabel('Kategori', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE (gün)', fontsize=12, fontweight='bold')
ax.set_title('Kategori Bazlı MAE Karşılaştırması', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_results['Kategori'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Bar üzerine değerler ekle
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'kategori_mae_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✅ {output_dir / 'kategori_mae_comparison.png'}")
plt.close()

# ============================================================================
# 8. MODELİ KAYDET
# ============================================================================
print("\n💾 Model kaydediliyor...")

model_info = {
    'model': log_model_baseline,
    'transformation': 'log1p',
    'baseline_metrics': {
        'rmse': baseline_rmse,
        'mae': baseline_mae,
        'r2': baseline_r2
    },
    'log_metrics': {
        'rmse': log_rmse,
        'mae': log_mae,
        'r2': log_r2
    },
    'improvements': {
        'rmse_improvement_pct': rmse_improvement,
        'mae_improvement_pct': mae_improvement,
        'r2_improvement_pct': r2_improvement
    },
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(output_dir / 'log_transform_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print(f"   ✅ Model kaydedildi: {output_dir / 'log_transform_model.pkl'}\n")

# ============================================================================
# 9. SONUÇ ÖZETİ
# ============================================================================
print("=" * 80)
print("SONUÇ ÖZETİ")
print("=" * 80)

if rmse_improvement > 0:
    print(f"✅ Log transformation BAŞARILI! RMSE {rmse_improvement:.1f}% iyileşti.")
else:
    print(f"⚠️  Log transformation beklenen etkiyi yaratmadı. RMSE {rmse_improvement:.1f}% değişim.")

if log_r2 > baseline_r2:
    print(f"✅ R² Score arttı: {baseline_r2:.4f} → {log_r2:.4f} (+{r2_improvement:.1f}%)")
else:
    print(f"⚠️  R² Score düştü: {baseline_r2:.4f} → {log_r2:.4f} ({r2_improvement:.1f}%)")

print("\n📌 En Büyük İyileşmeler (Kategorilere Göre):")
for idx, row in df_results.iterrows():
    mae_change = row['MAE_Fark']
    if mae_change > 0:
        print(f"   {row['Kategori']:8s}: MAE {mae_change:+.1f} gün iyileşti ✅")
    else:
        print(f"   {row['Kategori']:8s}: MAE {mae_change:+.1f} gün değişti ⚠️")

print("\n💡 ÖNERİ:")
if rmse_improvement > 5:  # %5+ iyileşme
    print("   Log-transformation modelini kullan! Belirgin iyileşme var.")
    print("   Sonraki adım: Hyperparameter tuning ile daha da iyileştir.")
elif rmse_improvement > 0:
    print("   Küçük iyileşme var. Kategori bazlı ayrı modeller dene.")
else:
    print("   Baseline modeli koru. Farklı stratejiler dene (ensemble, kategori bazlı modeller).")

print("\n" + "=" * 80)
print(f"✅ ANALİZ TAMAMLANDI! Tüm çıktılar: {output_dir}/")
print("=" * 80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
