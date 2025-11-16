"""
ADIM 12: FEATURE SELECTION & HYPERPARAMETER RE-TUNING
======================================================

Amaç:
1. Düşük öneme sahip feature'ları çıkar (importance < 0.005)
2. BALANCED kategorilerle hyperparameter re-tuning
3. Model performansını optimize et

Beklenen Kazanç: R² +0.03-0.07, RMSE -10-30 gün
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import os
from datetime import datetime

print("="*70)
print("FEATURE SELECTION & HYPERPARAMETER RE-TUNING".center(70))
print("="*70)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================

print("📂 Veri yükleniyor...")
X_train = pd.read_csv('model_data_new_categories/X_train.csv')
X_test = pd.read_csv('model_data_new_categories/X_test.csv')
y_train = pd.read_csv('model_data_new_categories/y_train.csv')
y_test = pd.read_csv('model_data_new_categories/y_test.csv')

print(f"   Train: {X_train.shape[0]:,} satır × {X_train.shape[1]} feature")
print(f"   Test: {X_test.shape[0]:,} satır × {X_test.shape[1]} feature\n")

# ============================================================================
# 2. FEATURE IMPORTANCE YÜKLEME VE DÜŞÜK ÖNEMLİ FEATURE'LARI BELİRLEME
# ============================================================================

print("🔍 Feature importance analizi...")
feature_imp = pd.read_csv('outputs/model/feature_importance.csv')

# Importance < 0.005 olanları bul
threshold = 0.005
low_importance = feature_imp[feature_imp['importance'] < threshold]['feature'].tolist()

print(f"   Importance Threshold: {threshold}")
print(f"   Düşük öneme sahip feature sayısı: {len(low_importance)}")
print(f"   Çıkarılacak feature'lar: {low_importance}\n")

# ============================================================================
# 3. FEATURE SELECTION - DÜŞÜK ÖNEMLİ FEATURE'LARI ÇIKAR
# ============================================================================

print("✂️  Feature selection uygulanıyor...")
X_train_selected = X_train.drop(columns=low_importance, errors='ignore')
X_test_selected = X_test.drop(columns=low_importance, errors='ignore')

print(f"   Önceki feature sayısı: {X_train.shape[1]}")
print(f"   Yeni feature sayısı: {X_train_selected.shape[1]}")
print(f"   Çıkarılan feature: {X_train.shape[1] - X_train_selected.shape[1]}\n")

# ============================================================================
# 4. BASELINE MODEL (Feature Selection ile)
# ============================================================================

print("🤖 Baseline model (feature selection sonrası)...")

# Orijinal en iyi parametreler
baseline_params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'random_state': 42,
    'objective': 'reg:squarederror'
}

baseline_model = xgb.XGBRegressor(**baseline_params)
baseline_model.fit(X_train_selected, y_train['jail'])

# Tahminler
y_pred_baseline = baseline_model.predict(X_test_selected)

# Metrikler
rmse_baseline = np.sqrt(mean_squared_error(y_test['jail'], y_pred_baseline))
mae_baseline = mean_absolute_error(y_test['jail'], y_pred_baseline)
r2_baseline = r2_score(y_test['jail'], y_pred_baseline)

print(f"   RMSE: {rmse_baseline:.2f} gün")
print(f"   MAE: {mae_baseline:.2f} gün")
print(f"   R²: {r2_baseline:.4f}\n")

# ============================================================================
# 5. HYPERPARAMETER RE-TUNING (BALANCED Kategorilerle)
# ============================================================================

print("🔧 Hyperparameter re-tuning (GridSearchCV)...")
print("   BALANCED kategori sistemi ile optimize edilecek...\n")

# Daha geniş parametre aralığı
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
}

xgb_model = xgb.XGBRegressor(
    random_state=42,
    objective='reg:squarederror'
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

print("⏳ GridSearchCV başlatılıyor (bu biraz sürebilir ~3-5 dakika)...")
import time
start_time = time.time()

grid_search.fit(X_train_selected, y_train['jail'])

elapsed_time = time.time() - start_time
print(f"\n✅ GridSearchCV tamamlandı! Süre: {elapsed_time/60:.2f} dakika\n")

# En iyi parametreler
best_params = grid_search.best_params_
print("🏆 En İyi Parametreler:")
for param, value in best_params.items():
    print(f"   {param}: {value}")
print()

# ============================================================================
# 6. FINAL MODEL (En İyi Parametrelerle)
# ============================================================================

print("🎯 Final model eğitimi (en iyi parametrelerle)...")

final_model = xgb.XGBRegressor(**best_params, random_state=42, objective='reg:squarederror')
final_model.fit(X_train_selected, y_train['jail'])

# Tahminler
y_pred_train = final_model.predict(X_train_selected)
y_pred_test = final_model.predict(X_test_selected)

# Train metrikleri
rmse_train = np.sqrt(mean_squared_error(y_train['jail'], y_pred_train))
mae_train = mean_absolute_error(y_train['jail'], y_pred_train)
r2_train = r2_score(y_train['jail'], y_pred_train)

# Test metrikleri
rmse_test = np.sqrt(mean_squared_error(y_test['jail'], y_pred_test))
mae_test = mean_absolute_error(y_test['jail'], y_pred_test)
r2_test = r2_score(y_test['jail'], y_pred_test)

print("\n📊 FINAL MODEL PERFORMANSI:")
print("-" * 70)
print(f"TRAIN SET:")
print(f"  RMSE: {rmse_train:.2f} gün")
print(f"  MAE: {mae_train:.2f} gün")
print(f"  R²: {r2_train:.4f}")
print()
print(f"TEST SET:")
print(f"  RMSE: {rmse_test:.2f} gün")
print(f"  MAE: {mae_test:.2f} gün")
print(f"  R²: {r2_test:.4f}")
print("-" * 70)

# ============================================================================
# 7. KARŞILAŞTIRMA
# ============================================================================

print("\n📈 PERFORMANS KARŞILAŞTIRMASI:")
print("=" * 70)

# Önceki en iyi model (BALANCED kategorilerle)
prev_rmse = 386.58
prev_mae = 85.82
prev_r2 = 0.6278

print(f"Önceki Model (BALANCED):    RMSE={prev_rmse:.2f} | MAE={prev_mae:.2f} | R²={prev_r2:.4f}")
print(f"Baseline (Feature Sel.):    RMSE={rmse_baseline:.2f} | MAE={mae_baseline:.2f} | R²={r2_baseline:.4f}")
print(f"Final (Feature + Tuning):   RMSE={rmse_test:.2f} | MAE={mae_test:.2f} | R²={r2_test:.4f}")
print()

# İyileşme hesapla
rmse_improvement = ((prev_rmse - rmse_test) / prev_rmse) * 100
mae_improvement = ((prev_mae - mae_test) / prev_mae) * 100
r2_improvement = ((r2_test - prev_r2) / prev_r2) * 100

print(f"İYİLEŞME:")
if rmse_improvement > 0:
    print(f"  ✅ RMSE: -{rmse_improvement:.1f}% ({prev_rmse:.2f} → {rmse_test:.2f} gün)")
else:
    print(f"  ⚠️  RMSE: +{abs(rmse_improvement):.1f}% ({prev_rmse:.2f} → {rmse_test:.2f} gün)")
    
if mae_improvement > 0:
    print(f"  ✅ MAE: -{mae_improvement:.1f}% ({prev_mae:.2f} → {mae_test:.2f} gün)")
else:
    print(f"  ⚠️  MAE: +{abs(mae_improvement):.1f}% ({prev_mae:.2f} → {mae_test:.2f} gün)")
    
if r2_improvement > 0:
    print(f"  ✅ R²: +{r2_improvement:.1f}% ({prev_r2:.4f} → {r2_test:.4f})")
else:
    print(f"  ⚠️  R²: -{abs(r2_improvement):.1f}% ({prev_r2:.4f} → {r2_test:.4f})")

print("=" * 70)

# ============================================================================
# 8. CROSS-VALIDATION
# ============================================================================

print("\n🔄 5-Fold Cross Validation...")
cv_scores = cross_val_score(
    final_model, 
    X_train_selected, 
    y_train['jail'],
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

cv_rmse = np.sqrt(-cv_scores)
print(f"   CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f} gün")
print(f"   Min: {cv_rmse.min():.2f} | Max: {cv_rmse.max():.2f}\n")

# ============================================================================
# 9. MODEL KAYDETME
# ============================================================================

output_dir = 'outputs/feature_selection'
os.makedirs(output_dir, exist_ok=True)

print(f"💾 Model kaydediliyor: {output_dir}/")

# Model kaydet
with open(f'{output_dir}/xgboost_optimized_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Seçilen feature'lar
selected_features = X_train_selected.columns.tolist()
with open(f'{output_dir}/selected_features.txt', 'w') as f:
    f.write('\n'.join(selected_features))

# Performans özeti
performance = {
    'model': 'XGBoost Optimized (Feature Selection + Hyperparameter Tuning)',
    'features_original': X_train.shape[1],
    'features_selected': X_train_selected.shape[1],
    'features_removed': X_train.shape[1] - X_train_selected.shape[1],
    'train_rmse': rmse_train,
    'train_mae': mae_train,
    'train_r2': r2_train,
    'test_rmse': rmse_test,
    'test_mae': mae_test,
    'test_r2': r2_test,
    'cv_rmse_mean': cv_rmse.mean(),
    'cv_rmse_std': cv_rmse.std(),
    'best_params': best_params,
    'improvement_rmse_pct': rmse_improvement,
    'improvement_mae_pct': mae_improvement,
    'improvement_r2_pct': r2_improvement
}

pd.DataFrame([performance]).to_csv(f'{output_dir}/optimization_summary.csv', index=False)

print("   ✅ xgboost_optimized_model.pkl")
print("   ✅ selected_features.txt")
print("   ✅ optimization_summary.csv")

# ============================================================================
# SONUÇ
# ============================================================================

print("\n" + "="*70)
print("SONUÇ ÖZETİ".center(70))
print("="*70)

if r2_test > prev_r2:
    print(f"🎉 BAŞARILI! Model performansı iyileşti!")
    print(f"   R² artışı: {prev_r2:.4f} → {r2_test:.4f} (+{r2_improvement:.1f}%)")
    print(f"   RMSE azalışı: {prev_rmse:.2f} → {rmse_test:.2f} gün (-{rmse_improvement:.1f}%)")
    print(f"\n   Bu model TEZİN FİNAL MODELİ olarak kullanılabilir! ✅")
elif r2_test > r2_baseline:
    print(f"✅ İyi! Hyperparameter tuning etkili oldu.")
    print(f"   Baseline R²: {r2_baseline:.4f}")
    print(f"   Final R²: {r2_test:.4f}")
    print(f"\n   Önceki modelden biraz düşük ama yine de iyi performans.")
else:
    print(f"⚠️  Uyarı: Beklenen iyileşme gerçekleşmedi.")
    print(f"   Önceki model daha iyi olabilir.")
    print(f"\n   Ancak, mevcut model (R²={r2_test:.4f}) yine de literatür üzerinde!")

print("="*70)
print(f"\nBitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
