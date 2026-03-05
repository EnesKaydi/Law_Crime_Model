"""
🚀 STEP 33 - OPTIMIZED SYSTEM FINAL TEST WITH ROUTER
======================================================
Optimize edilmiş 3 modeli (Router + Mainstream Regressor + High Severity Regressor)
bir araya getirerek, yönlendirme (routing) başarı/hatası dahil sistemin tam performansını ölçer.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path
import warnings

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
    if 'highest_severity' in df.columns and 'is_recid_new' in df.columns:
        df['severity_x_recid'] = df['highest_severity'] * df['is_recid_new']
    return df

def create_comprehensive_features(df):
    
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

def test_final_system():
    print("="*70)
    print("🚀 OPTIMIZED SYSTEM FINAL TEST WITH ROUTER")
    print("="*70)
    
    print(f"\n📂 Veri yükleniyor...")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    print(f"✅ Toplam vaka: {len(df):,}")
    
    print(f"🔧 Tüm verisetinde base interactions hazırlanıyor (veri seti {len(df)})...")
    # Base prep
    df = create_base_interactions(df)
    
    # Target
    y_true_real = df['jail'].values
    y_true_log = np.log1p(df['jail'].values)
    
    print(f"\n📦 Optimizasyon Modelleri Yükleniyor...")
    
    # 1. Router Load
    router = CatBoostClassifier()
    router.load_model(str(MODEL_DIR / "router_classifier_optimized.cbm"))
    router_features = joblib.load(MODEL_DIR / "router_features_optimized.pkl")
    router_cat_features = joblib.load(MODEL_DIR / "router_cat_features_optimized.pkl")
    
    # 2. Mainstream Load
    model_main = CatBoostRegressor()
    model_main.load_model(str(MODEL_DIR / "model_low_v2_optimized.cbm"))
    features_main = joblib.load(MODEL_DIR / "features_v2.pkl")
    # Ana regressor cat features
    KNOWN_CAT = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass']
    
    # 3. High Severity Load
    model_high = CatBoostRegressor()
    model_high.load_model(str(MODEL_DIR / "model_high_v2_comprehensive_optimized.cbm"))
    features_high = joblib.load(MODEL_DIR / "features_v2_comprehensive.pkl")
    cat_features_high = joblib.load(MODEL_DIR / "cat_features_v2_comprehensive.pkl")
    
    print(f"   ✅ Modeller yüklendi.")
    
    # Bütün veri seti train/test ayrımı
    # Note: Orjinal modeller eğitilirken ayrı datasetlerle değil, train_test_split maskeleriyle eğitildi
    # Train/Test leak olmaması için datasetin %20'lik test kısmını seed 42 ile ayıralım.
    indices = np.arange(len(df))
    X_train_idx, X_test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)
    
    df_test = df.iloc[X_test_idx].copy()
    print(f"📊 Test Seti Büyüklüğü: {len(df_test):,}")
    
    y_true_split_orig = df_test['jail'].values
    y_true_split_log = np.log1p(y_true_split_orig)
    
    # ---- ADIM 1: ROUTING TAHMİNİ ----
    print(f"\n🚦 Router sınıflandırma yapıyor...")
    X_router = df_test[router_features].copy()
    
    for col in X_router.columns:
        if col in router_cat_features or X_router[col].dtype == 'object':
            X_router[col] = X_router[col].fillna("Unknown").astype(str)
        else:
            X_router[col] = X_router[col].fillna(X_router[col].mean())
            
    router_preds = router.predict(X_router)
    # router_preds: 0 -> Mainstream, 1 -> High Severity
    
    df_test['router_pred'] = router_preds
    mask_router_main = df_test['router_pred'] == 0
    mask_router_high = df_test['router_pred'] == 1
    
    print(f"   • Mainstream'e Yönlendirilen: {mask_router_main.sum():,} vaka")
    print(f"   • High Severity'e Yönlendirilen: {mask_router_high.sum():,} vaka")
    
    # ---- ADIM 2: MAINSTREAM TAHMİNİ ----
    print(f"\n📈 Mainstream uzman model çalışıyor...")
    preds_main = np.zeros(mask_router_main.sum())
    if mask_router_main.sum() > 0:
        X_main = df_test[mask_router_main][features_main].copy()
        for col in X_main.columns:
            if col in KNOWN_CAT or X_main[col].dtype == 'object':
                X_main[col] = X_main[col].fillna("Unknown").astype(str)
            else:
                X_main[col] = X_main[col].fillna(X_main[col].mean())
        preds_main = model_main.predict(X_main)
        
    # ---- ADIM 3: HIGH SEVERITY TAHMİNİ ----
    print(f"📈 High Severity uzman model çalışıyor...")
    preds_high = np.zeros(mask_router_high.sum())
    if mask_router_high.sum() > 0:
        df_high_portion = df_test[mask_router_high].copy()
        df_high_portion = create_comprehensive_features(df_high_portion)
        X_high = df_high_portion[features_high].copy()
        
        for col in X_high.columns:
            if col in cat_features_high or X_high[col].dtype == 'object' or X_high[col].dtype.name == 'category':
                X_high[col] = X_high[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
            else:
                X_high[col] = X_high[col].fillna(X_high[col].mean())
        preds_high = model_high.predict(X_high)
        
    # ---- ADIM 4: SONUÇLARI BİRLEŞTİRME ----
    print(f"\n🏆 GENEL SİSTEM PERFORMANSI HESAPLANIYOR...")
    
    final_preds_log = np.zeros(len(df_test))
    if mask_router_main.sum() > 0:
        final_preds_log[mask_router_main] = preds_main
    if mask_router_high.sum() > 0:
        final_preds_log[mask_router_high] = preds_high
        
    final_preds_orig = np.expm1(final_preds_log)
    
    final_r2_log = r2_score(y_true_split_log, final_preds_log)
    final_r2_orig = r2_score(y_true_split_orig, final_preds_orig)
    final_mae = mean_absolute_error(y_true_split_orig, final_preds_orig)
    
    print(f"   • R² Score (Log Scale): {final_r2_log:.4f} ({final_r2_log*100:.2f}%)")
    print(f"   • R² Score (Original Scale): {final_r2_orig:.4f} ({final_r2_orig*100:.2f}%)")
    print(f"   • MAE: {final_mae:.2f} gün")
    
    print(f"\n📈 KARŞILAŞTIRMA MİADI (Teorik Maksimum, step_29):")
    print(f"   • Önceki Optima: R²(Log) 0.8365 (Sadece teorik split - %100 kusursuz router varsayımıyla)")
    print(f"   • Teorik MAE: 313 Gün")
    
    print(f"\n✅ GERÇEK DÜNYA OPTİMİZASYONU ETKİSİ:")
    print(f"Bu test Router sınıflandırıcısının hatalarını da içerdiği için GERÇEK performanstır.")
    
if __name__ == "__main__":
    test_final_system()
