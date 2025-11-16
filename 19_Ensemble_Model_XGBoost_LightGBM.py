"""
ADIM 13: ENSEMBLE MODEL - XGBoost + LightGBM
=============================================

Amaç:
1. XGBoost ve LightGBM modellerini ayrı ayrı eğit
2. Voting ensemble ile tahminleri birleştir
3. Performans artışını değerlendir

Beklenen Kazanç: R² +0.03-0.07, RMSE -15-30 gün
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
from datetime import datetime

print("="*70)
print("ENSEMBLE MODEL - XGBoost + LightGBM".center(70))
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
# 2. MODEL 1: XGBoost (Önceki En İyi Parametreler)
# ============================================================================

print("🤖 MODEL 1: XGBoost eğitimi...")

xgb_params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'n_jobs': -1
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train['jail'])

# XGBoost tahminleri
y_pred_xgb_test = xgb_model.predict(X_test)

# XGBoost metrikleri
rmse_xgb = np.sqrt(mean_squared_error(y_test['jail'], y_pred_xgb_test))
mae_xgb = mean_absolute_error(y_test['jail'], y_pred_xgb_test)
r2_xgb = r2_score(y_test['jail'], y_pred_xgb_test)

print(f"   RMSE: {rmse_xgb:.2f} gün")
print(f"   MAE: {mae_xgb:.2f} gün")
print(f"   R²: {r2_xgb:.4f}\n")

# ============================================================================
# 3. MODEL 2: LightGBM
# ============================================================================

print("🤖 MODEL 2: LightGBM eğitimi...")

lgb_params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'random_state': 42,
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train['jail'])

# LightGBM tahminleri
y_pred_lgb_test = lgb_model.predict(X_test)

# LightGBM metrikleri
rmse_lgb = np.sqrt(mean_squared_error(y_test['jail'], y_pred_lgb_test))
mae_lgb = mean_absolute_error(y_test['jail'], y_pred_lgb_test)
r2_lgb = r2_score(y_test['jail'], y_pred_lgb_test)

print(f"   RMSE: {rmse_lgb:.2f} gün")
print(f"   MAE: {mae_lgb:.2f} gün")
print(f"   R²: {r2_lgb:.4f}\n")

# ============================================================================
# 4. ENSEMBLE - Voting (Ortalama)
# ============================================================================

print("🎯 ENSEMBLE MODEL (Weighted Average)...")

# Basit ortalama (eşit ağırlık)
y_pred_ensemble_simple = (y_pred_xgb_test + y_pred_lgb_test) / 2

# Simple ensemble metrikleri
rmse_ens_simple = np.sqrt(mean_squared_error(y_test['jail'], y_pred_ensemble_simple))
mae_ens_simple = mean_absolute_error(y_test['jail'], y_pred_ensemble_simple)
r2_ens_simple = r2_score(y_test['jail'], y_pred_ensemble_simple)

print(f"   Simple Average:")
print(f"   RMSE: {rmse_ens_simple:.2f} gün")
print(f"   MAE: {mae_ens_simple:.2f} gün")
print(f"   R²: {r2_ens_simple:.4f}\n")

# Weighted average (XGBoost biraz daha yüksek ağırlık)
weight_xgb = 0.6
weight_lgb = 0.4

y_pred_ensemble_weighted = (weight_xgb * y_pred_xgb_test + weight_lgb * y_pred_lgb_test)

# Weighted ensemble metrikleri
rmse_ens_weighted = np.sqrt(mean_squared_error(y_test['jail'], y_pred_ensemble_weighted))
mae_ens_weighted = mean_absolute_error(y_test['jail'], y_pred_ensemble_weighted)
r2_ens_weighted = r2_score(y_test['jail'], y_pred_ensemble_weighted)

print(f"   Weighted Average (XGB:0.6, LGB:0.4):")
print(f"   RMSE: {rmse_ens_weighted:.2f} gün")
print(f"   MAE: {mae_ens_weighted:.2f} gün")
print(f"   R²: {r2_ens_weighted:.4f}\n")

# ============================================================================
# 5. PERFORMANS KARŞILAŞTIRMASI
# ============================================================================

print("📈 PERFORMANS KARŞILAŞTIRMASI:")
print("=" * 70)

# Önceki en iyi model (BALANCED kategorilerle - tek XGBoost)
prev_rmse = 386.58
prev_mae = 85.82
prev_r2 = 0.6278

print(f"Önceki Model (XGBoost):     RMSE={prev_rmse:.2f} | MAE={prev_mae:.2f} | R²={prev_r2:.4f}")
print(f"XGBoost (yeniden):          RMSE={rmse_xgb:.2f} | MAE={mae_xgb:.2f} | R²={r2_xgb:.4f}")
print(f"LightGBM:                   RMSE={rmse_lgb:.2f} | MAE={mae_lgb:.2f} | R²={r2_lgb:.4f}")
print(f"Ensemble (Simple):          RMSE={rmse_ens_simple:.2f} | MAE={mae_ens_simple:.2f} | R²={r2_ens_simple:.4f}")
print(f"Ensemble (Weighted):        RMSE={rmse_ens_weighted:.2f} | MAE={mae_ens_weighted:.2f} | R²={r2_ens_weighted:.4f}")
print()

# En iyi modeli belirle
models = {
    'Önceki XGBoost': (prev_rmse, prev_mae, prev_r2),
    'XGBoost (yeni)': (rmse_xgb, mae_xgb, r2_xgb),
    'LightGBM': (rmse_lgb, mae_lgb, r2_lgb),
    'Ensemble Simple': (rmse_ens_simple, mae_ens_simple, r2_ens_simple),
    'Ensemble Weighted': (rmse_ens_weighted, mae_ens_weighted, r2_ens_weighted)
}

# R² en yüksek olan
best_model = max(models.items(), key=lambda x: x[1][2])
best_name, (best_rmse, best_mae, best_r2) = best_model

print(f"🏆 EN İYİ MODEL: {best_name}")
print(f"   RMSE: {best_rmse:.2f} gün")
print(f"   MAE: {best_mae:.2f} gün")
print(f"   R²: {best_r2:.4f}")

# İyileşme hesapla
rmse_improvement = ((prev_rmse - best_rmse) / prev_rmse) * 100
mae_improvement = ((prev_mae - best_mae) / prev_mae) * 100
r2_improvement = ((best_r2 - prev_r2) / prev_r2) * 100

print()
print("İYİLEŞME (Önceki modele göre):")
if rmse_improvement > 0:
    print(f"  ✅ RMSE: -{rmse_improvement:.1f}% ({prev_rmse:.2f} → {best_rmse:.2f} gün)")
else:
    print(f"  ⚠️  RMSE: +{abs(rmse_improvement):.1f}% ({prev_rmse:.2f} → {best_rmse:.2f} gün)")
    
if mae_improvement > 0:
    print(f"  ✅ MAE: -{mae_improvement:.1f}% ({prev_mae:.2f} → {best_mae:.2f} gün)")
else:
    print(f"  ⚠️  MAE: +{abs(mae_improvement):.1f}% ({prev_mae:.2f} → {best_mae:.2f} gün)")
    
if r2_improvement > 0:
    print(f"  ✅ R²: +{r2_improvement:.1f}% ({prev_r2:.4f} → {best_r2:.4f})")
else:
    print(f"  ⚠️  R²: -{abs(r2_improvement):.1f}% ({prev_r2:.4f} → {best_r2:.4f})")

print("=" * 70)

# ============================================================================
# 6. MODEL KAYDETME
# ============================================================================

output_dir = 'outputs/ensemble'
os.makedirs(output_dir, exist_ok=True)

print(f"\n💾 Modeller kaydediliyor: {output_dir}/")

# XGBoost kaydet
with open(f'{output_dir}/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# LightGBM kaydet
with open(f'{output_dir}/lightgbm_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)

# Performans özeti
performance = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'Ensemble_Simple', 'Ensemble_Weighted'],
    'RMSE': [rmse_xgb, rmse_lgb, rmse_ens_simple, rmse_ens_weighted],
    'MAE': [mae_xgb, mae_lgb, mae_ens_simple, mae_ens_weighted],
    'R2': [r2_xgb, r2_lgb, r2_ens_simple, r2_ens_weighted]
})

performance.to_csv(f'{output_dir}/ensemble_performance.csv', index=False)

print("   ✅ xgboost_model.pkl")
print("   ✅ lightgbm_model.pkl")
print("   ✅ ensemble_performance.csv")

# ============================================================================
# SONUÇ
# ============================================================================

print("\n" + "="*70)
print("SONUÇ ÖZETİ".center(70))
print("="*70)

if best_r2 > prev_r2:
    print(f"🎉 BAŞARILI! Ensemble model performansı iyileştirdi!")
    print(f"   En İyi Model: {best_name}")
    print(f"   R² artışı: {prev_r2:.4f} → {best_r2:.4f} (+{r2_improvement:.1f}%)")
    print(f"   RMSE azalışı: {prev_rmse:.2f} → {best_rmse:.2f} gün (-{rmse_improvement:.1f}%)")
    print(f"\n   Bu model TEZİN FİNAL MODELİ olarak kullanılabilir! ✅")
elif best_r2 >= prev_r2 - 0.001:  # Çok küçük fark
    print(f"✅ İyi! Ensemble model önceki model ile eşdeğer performans gösteriyor.")
    print(f"   R² farkı çok küçük: {abs(prev_r2 - best_r2):.4f}")
    print(f"\n   Önceki model tutulabilir veya ensemble kullanılabilir.")
else:
    print(f"⚠️  Ensemble model beklenen iyileştirmeyi vermedi.")
    print(f"   Önceki tek XGBoost modeli daha iyi performans gösteriyor.")
    print(f"\n   Önceki model (R²={prev_r2:.4f}) FİNAL MODEL olarak kullanılmalı.")

print("="*70)
print(f"\nBitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
