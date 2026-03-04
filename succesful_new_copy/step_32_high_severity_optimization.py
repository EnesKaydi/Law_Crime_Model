"""
🚀 STEP 32 - HIGH SEVERITY MODEL OPTIMIZATION
==============================================
High Severity modeli (>3000 gün cezalar) için Optuna kullanarak
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

def create_comprehensive_features(df):
    df = create_base_interactions(df)
    
    if 'judge_id' in df.columns:
        df['judge_mean_sentence'] = df.groupby('judge_id')['jail'].transform('mean')
        df['judge_median_sentence'] = df.groupby('judge_id')['jail'].transform('median')
        df['judge_std_sentence'] = df.groupby('judge_id')['jail'].transform('std').fillna(0)
        df['judge_min_sentence'] = df.groupby('judge_id')['jail'].transform('min')
        df['judge_max_sentence'] = df.groupby('judge_id')['jail'].transform('max')
        df['judge_case_count'] = df.groupby('judge_id')['jail'].transform('count')
    
    if 'county' in df.columns:
        df['county_mean_sentence'] = df.groupby('county')['jail'].transform('mean')
        df['county_median_sentence'] = df.groupby('county')['jail'].transform('median')
        df['county_std_sentence'] = df.groupby('county')['jail'].transform('std').fillna(0)
    
    if 'wcisclass' in df.columns:
        df['wcisclass_mean_sentence'] = df.groupby('wcisclass')['jail'].transform('mean')
        df['wcisclass_median_sentence'] = df.groupby('wcisclass')['jail'].transform('median')
        df['wcisclass_std_sentence'] = df.groupby('wcisclass')['jail'].transform('std').fillna(0)
    
    if 'case_type' in df.columns:
        df['case_type_mean_sentence'] = df.groupby('case_type')['jail'].transform('mean')
        df['case_type_median_sentence'] = df.groupby('case_type')['jail'].transform('median')
    
    if 'judge_id' in df.columns and 'wcisclass' in df.columns:
        df['judge_crime_combo'] = df['judge_id'].astype(str) + '_' + df['wcisclass'].astype(str)
        df['judge_crime_mean'] = df.groupby('judge_crime_combo')['jail'].transform('mean')
    
    if 'judge_id' in df.columns and 'county' in df.columns:
        df['judge_county_combo'] = df['judge_id'].astype(str) + '_' + df['county'].astype(str)
        df['judge_county_mean'] = df.groupby('judge_county_combo')['jail'].transform('mean')
    
    if 'highest_severity' in df.columns and 'is_recid_new' in df.columns:
        df['severity_x_recid'] = df['highest_severity'] * df['is_recid_new']
    
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_ratio'] = age_j / (age_o + 1)
        df['age_product'] = age_j * age_o
    
    prior_cols = ['prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic']
    available_prior = [c for c in prior_cols if c in df.columns]
    if available_prior:
        df['total_prior'] = df[available_prior].fillna(0).sum(axis=1)
        if 'violent_crime' in df.columns:
            df['violent_x_prior'] = df['violent_crime'] * df['total_prior']
        if 'highest_severity' in df.columns:
            df['severity_x_total_prior'] = df['highest_severity'] * df['total_prior']
            df['severity_per_prior'] = df['highest_severity'] / (df['total_prior'] + 1)
    
    if 'age_offense' in df.columns:
        df['age_bin'] = pd.cut(df['age_offense'].fillna(df['age_offense'].mean()), 
                               bins=[0, 25, 35, 45, 55, 100], 
                               labels=['very_young', 'young', 'middle', 'mature', 'senior'])
    
    if 'highest_severity' in df.columns:
        df['severity_bin'] = pd.cut(df['highest_severity'], 
                                     bins=[0, 5, 10, 15, 20], 
                                     labels=['low', 'medium', 'high', 'very_high'])
    
    if 'year' in df.columns:
        df['years_since_2000'] = df['year'] - 2000
        df['year_squared'] = df['years_since_2000'] ** 2
        df['decade'] = (df['year'] // 10) * 10
    
    if 'highest_severity' in df.columns:
        df['severity_squared'] = df['highest_severity'] ** 2
        df['severity_cubed'] = df['highest_severity'] ** 3
    
    if 'age_offense' in df.columns:
        age_filled = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_squared'] = age_filled ** 2
    
    risk_components = []
    if 'violent_crime' in df.columns:
        risk_components.append(df['violent_crime'] * 3)
    if 'is_recid_new' in df.columns:
        risk_components.append(df['is_recid_new'] * 2)
    if 'highest_severity' in df.columns:
        risk_components.append(df['highest_severity'] / 10)
    
    if risk_components:
        df['composite_risk_score'] = sum(risk_components)
    
    return df

def optimize_high_severity():
    print("="*70)
    print("🚀 HIGH SEVERITY MODEL OPTIMIZATION (Optuna)")
    print("="*70)
    
    # 1. Veri Yükle ve Filtrele
    print(f"📂 Veri yükleniyor...")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    df_high = df[df['jail'] > THRESHOLD].copy()
    print(f"✅ High Severity Vaka Sayısı: {len(df_high):,}")
    
    # 2. Features Hazırla
    df_high = create_comprehensive_features(df_high)
    
    features = joblib.load(MODEL_DIR / "features_v2_comprehensive.pkl")
    X = df_high[features].copy()
    y = np.log1p(df_high['jail'])
    
    # 3. Kategorik Değişkenler
    cat_features = joblib.load(MODEL_DIR / "cat_features_v2_comprehensive.pkl")
    
    for col in X.columns:
        if col in cat_features or X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
            if col not in cat_features:
                cat_features.append(col)
                
    for col in X.columns:
        if col not in cat_features:
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
    final_model.save_model(str(MODEL_DIR / "model_high_v2_comprehensive_optimized.cbm"))
    print(f"\n💾 Optimize Edilmiş High Severity Model Kaydedildi: {MODEL_DIR}")

if __name__ == "__main__":
    optimize_high_severity()
