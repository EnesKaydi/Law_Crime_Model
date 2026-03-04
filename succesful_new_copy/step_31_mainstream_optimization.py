"""
🚀 STEP 31 - MAINSTREAM MODEL OPTIMIZATION
===========================================
Mainstream modeli (<=3000 gün cezalar) için Optuna kullanarak
hiperparametre optimizasyonu gerçekleştirilmesi.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path
import warnings
import optuna

warnings.filterwarnings('ignore')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("../model_data_v2_interactions")
THRESHOLD = 3000
RANDOM_STATE = 42

def create_base_interactions(df):
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
    return df

def optimize_mainstream():
    print("="*70)
    print("🚀 MAINSTREAM MODEL OPTIMIZATION (Optuna)")
    print("="*70)
    
    # 1. Veri Yükle ve Filtrele
    print(f"📂 Veri yükleniyor...")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    df_mainstream = df[df['jail'] <= THRESHOLD].copy()
    print(f"✅ Mainstream Vaka Sayısı: {len(df_mainstream):,}")
    
    # 2. Features Hazırla
    df_mainstream = create_base_interactions(df_mainstream)
    
    features = joblib.load(MODEL_DIR / "features_v2.pkl")
    X = df_mainstream[features].copy()
    y = np.log1p(df_mainstream['jail'])
    
    # 3. Kategorik Değişkenler
    KNOWN_CAT = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass']
    cat_features = []
    for col in X.columns:
        if col in KNOWN_CAT or X[col].dtype == 'object':
            X[col] = X[col].fillna("Unknown").astype(str)
            cat_features.append(col)
            
    for col in X.columns:
        if X[col].dtype != 'object':
            X[col] = X[col].fillna(X[col].mean())
            
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # 5. Optuna Objective
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_seed': RANDOM_STATE,
            'verbose': 0,
            'eval_metric': 'RMSE'
        }
        
        model = CatBoostRegressor(**params, cat_features=cat_features)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)
        
        preds = model.predict(X_test)
        return r2_score(y_test, preds) # R2 maximize
        
    print("\n🔍 Optuna ile Hyperparameter Tuning Başlıyor...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100) # Gece boyu çalışma (100 deneme)
    
    print("\n🏆 En İyi Parametreler:")
    print(study.best_params)
    print(f"🥇 En İyi R² Score: {study.best_value:.4f}")
    
    # 6. Final Model
    print("\n🚀 Final Model Eğitimi")
    best_params = study.best_params
    best_params['random_seed'] = RANDOM_STATE
    best_params['verbose'] = 100
    
    final_model = CatBoostRegressor(**best_params, cat_features=cat_features)
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    # Değerlendirme
    preds = final_model.predict(X_test)
    preds_orig = np.expm1(preds)
    y_test_orig = np.expm1(y_test)
    
    r2_log = r2_score(y_test, preds)
    r2_orig = r2_score(y_test_orig, preds_orig)
    mae = mean_absolute_error(y_test_orig, preds_orig)
    
    print(f"\n📈 SONUÇLAR:")
    print(f"   • R² Score (Log): {r2_log:.4f}")
    print(f"   • R² Score (Orig): {r2_orig:.4f}")
    print(f"   • MAE: {mae:.2f} gün")
    
    # 7. Kaydet
    final_model.save_model(str(MODEL_DIR / "model_low_v2_optimized.cbm"))
    print(f"\n💾 Optimize Edilmiş Mainstream Model Kaydedildi: {MODEL_DIR}")

if __name__ == "__main__":
    optimize_mainstream()
