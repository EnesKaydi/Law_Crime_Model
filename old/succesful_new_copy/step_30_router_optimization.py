"""
🚀 STEP 30 - ROUTER OPTIMIZATION (Feature Engineering + Hyperparameter Tuning)
=============================================================================
Router modelinin davanın 3000 gün altı mı üstü mü olduğunu bilme oranını artırmak
için, hem regression modellerinde keşfettiğimiz feature'ları sınıflayıcıya
ekliyoruz hem de Optuna ile hiperparametre optimizasyonu yapıyoruz.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from pathlib import Path
import warnings
import optuna

warnings.filterwarnings('ignore')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("../model_data_v2_interactions")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD = 3000

def create_router_features(df):
    """Router için özel özellik mühendisliği (Interaction'lar eklenmiş)"""
    df = df.copy()
    
    # Base interactions
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
        
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
        
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
        
    # Group hedefleri yerine (data leak olmaması için target bazlı encoding kullanmıyoruz)
    # Suç kodu şiddet kombinasyonları
    if 'highest_severity' in df.columns and 'is_recid_new' in df.columns:
        df['severity_x_recid'] = df['highest_severity'] * df['is_recid_new']
        
    return df

def optimize_router():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    df['target_class'] = (df['jail'] > THRESHOLD).astype(int)
    
    print("\n📊 Sınıf Dağılımı (0: <=3000, 1: >3000):")
    print(df['target_class'].value_counts(normalize=True))
    
    # Mevcut Özellikler
    base_features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    base_features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    
    # Feature Engineering
    df = create_router_features(df)
    
    # Yeni feature'ları ekle
    new_features = ['severity_x_violent', 'age_gap', 'violent_recid', 'severity_x_recid']
    base_features.extend(new_features)
    
    available_features = [f for f in base_features if f in df.columns]
    
    X = df[available_features].copy()
    y = df['target_class']
    
    # Kategorik İşlemler
    cat_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("Unknown").astype(str)
            cat_features.append(col)
            
    for col in ['judge_id', 'county', 'zip']:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("Unknown")
            if col not in cat_features: cat_features.append(col)
            
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
            
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': 0,
            'eval_metric': 'F1'
        }
        
        model = CatBoostClassifier(**params, cat_features=cat_features)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)
        
        preds = model.predict(X_test)
        return f1_score(y_test, preds)

    print("\n🔍 Optuna ile Hyperparameter Tuning Başlıyor...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100) # Gece boyu çalışma (300 deneme)
    
    print("\n🏆 En İyi Parametreler:")
    print(study.best_params)
    print(f"🥇 En İyi F1 Score: {study.best_value:.4f}")
    
    # En iyi modelle tekrar eğitim
    print("\n🚀 Final Model Eğitimi (En İyi Parametrelerle)")
    best_params = study.best_params
    best_params['auto_class_weights'] = 'Balanced'
    best_params['random_seed'] = 42
    best_params['verbose'] = 100
    best_params['eval_metric'] = 'F1'
    
    final_model = CatBoostClassifier(**best_params, cat_features=cat_features)
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    # Değerlendirme
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n🔹 Final Accuracy: %{acc*100:.2f}")
    print(f"🔹 Final F1 Score: %{f1*100:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Kayıt
    final_model.save_model(str(MODEL_DIR / "router_classifier_optimized.cbm"))
    joblib.dump(available_features, MODEL_DIR / "router_features_optimized.pkl")
    joblib.dump(cat_features, MODEL_DIR / "router_cat_features_optimized.pkl")
    print(f"\n💾 Optimize Edilmiş Router Model Kaydedildi: {MODEL_DIR}")

if __name__ == "__main__":
    optimize_router()
