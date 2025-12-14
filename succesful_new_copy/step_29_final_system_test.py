"""
🚀 FINAL SYSTEM TEST - COMPREHENSIVE MODEL
===========================================

Comprehensive High Severity modelini sisteme entegre et ve GERÇEK performansı ölç!

step_16 gibi: Her iki modeli test setinde çalıştır, sonuçları birleştir, genel R² hesapla.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
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
    """Base interaction features (41 feature için)"""
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
    """Comprehensive features (75 feature için - sadece High Severity)"""
    # Base interactions
    df = create_base_interactions(df)
    
    # Groupby transforms
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
    
    # More interactions
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
    
    # Binning
    if 'age_offense' in df.columns:
        df['age_bin'] = pd.cut(df['age_offense'].fillna(df['age_offense'].mean()), 
                               bins=[0, 25, 35, 45, 55, 100], 
                               labels=['very_young', 'young', 'middle', 'mature', 'senior'])
    
    if 'highest_severity' in df.columns:
        df['severity_bin'] = pd.cut(df['highest_severity'], 
                                     bins=[0, 5, 10, 15, 20], 
                                     labels=['low', 'medium', 'high', 'very_high'])
    
    # Temporal
    if 'year' in df.columns:
        df['years_since_2000'] = df['year'] - 2000
        df['year_squared'] = df['years_since_2000'] ** 2
        df['decade'] = (df['year'] // 10) * 10
    
    # Polynomial
    if 'highest_severity' in df.columns:
        df['severity_squared'] = df['highest_severity'] ** 2
        df['severity_cubed'] = df['highest_severity'] ** 3
    
    if 'age_offense' in df.columns:
        age_filled = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_squared'] = age_filled ** 2
    
    # Risk score
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
    """Final sistem testi - step_16 mantığıyla"""
    print("="*70)
    print("🚀 FINAL SYSTEM TEST - COMPREHENSIVE MODEL")
    print("="*70)
    
    # 1. Veri yükle
    print(f"\n📂 Veri yükleniyor...")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    print(f"✅ Toplam vaka: {len(df):,}")
    
    # 2. Segmentlere ayır
    mask_mainstream = df['jail'] <= THRESHOLD
    mask_high = df['jail'] > THRESHOLD
    
    df_mainstream = df[mask_mainstream].copy()
    df_high = df[mask_high].copy()
    
    print(f"\n📊 Segmentler:")
    print(f"   • Mainstream: {len(df_mainstream):,} ({len(df_mainstream)/len(df)*100:.1f}%)")
    print(f"   • High Severity: {len(df_high):,} ({len(df_high)/len(df)*100:.1f}%)")
    
    # 3. Mainstream için features hazırla (41 feature)
    print(f"\n🔧 Mainstream features hazırlanıyor...")
    df_mainstream = create_base_interactions(df_mainstream)
    
    # 4. High Severity için features hazırla (75 feature)
    print(f"🔧 High Severity comprehensive features hazırlanıyor...")
    df_high = create_comprehensive_features(df_high)
    
    # 5. Modelleri yükle
    print(f"\n📦 Modeller yükleniyor...")
    
    model_mainstream = CatBoostRegressor()
    model_mainstream.load_model(str(MODEL_DIR / "model_low_v2.cbm"))
    print(f"   ✅ Mainstream Model")
    
    model_high = CatBoostRegressor()
    model_high.load_model(str(MODEL_DIR / "model_high_v2_comprehensive.cbm"))
    print(f"   ✅ High Severity Comprehensive Model")
    
    # 6. Feature listelerini yükle
    features_mainstream = joblib.load(MODEL_DIR / "features_v2.pkl")
    features_high = joblib.load(MODEL_DIR / "features_v2_comprehensive.pkl")
    cat_features_high = joblib.load(MODEL_DIR / "cat_features_v2_comprehensive.pkl")
    
    print(f"\n   • Mainstream features: {len(features_mainstream)}")
    print(f"   • High Severity features: {len(features_high)}")
    
    # 7. Test setleri oluştur
    print(f"\n🧪 Test setleri hazırlanıyor...")
    
    # Mainstream
    X_main = df_mainstream[features_mainstream].copy()
    y_main = np.log1p(df_mainstream['jail'])
    
    # Kategorik dönüşüm
    KNOWN_CAT = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass']
    for col in X_main.columns:
        if col in KNOWN_CAT or X_main[col].dtype == 'object':
            X_main[col] = X_main[col].fillna("Unknown").astype(str)
    for col in X_main.columns:
        if X_main[col].dtype != 'object':
            X_main[col] = X_main[col].fillna(X_main[col].mean())
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_main, y_main, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # High Severity
    X_high = df_high[features_high].copy()
    y_high = np.log1p(df_high['jail'])
    
    # Kategorik dönüşüm
    KNOWN_CAT_HIGH = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass', 
                      'judge_crime_combo', 'judge_county_combo', 'age_bin', 'severity_bin', 'decade']
    for col in X_high.columns:
        if col in KNOWN_CAT_HIGH or X_high[col].dtype == 'object' or X_high[col].dtype.name == 'category':
            X_high[col] = X_high[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
    for col in X_high.columns:
        if col not in cat_features_high:
            X_high[col] = X_high[col].fillna(X_high[col].mean())
    
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_high, y_high, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"   • Mainstream test: {len(X_test_m):,}")
    print(f"   • High Severity test: {len(X_test_h):,}")
    
    # 8. Tahminler
    print(f"\n🔮 Tahminler yapılıyor...")
    
    pred_main = model_mainstream.predict(X_test_m)
    pred_high = model_high.predict(X_test_h)
    
    # 9. Segment bazlı performans
    print(f"\n📊 SEGMENT BAZLI PERFORMANS:")
    
    r2_main = r2_score(y_test_m, pred_main)
    r2_high = r2_score(y_test_h, pred_high)
    
    print(f"   • Mainstream R²: {r2_main:.4f} ({r2_main*100:.2f}%)")
    print(f"   • High Severity R²: {r2_high:.4f} ({r2_high*100:.2f}%)")
    
    # 10. Genel sistem performansı (step_16 gibi)
    print(f"\n🏆 GENEL SİSTEM PERFORMANSI:")
    
    all_y_true_log = np.concatenate([y_test_m, y_test_h])
    all_y_pred_log = np.concatenate([pred_main, pred_high])
    all_y_true = np.expm1(all_y_true_log)
    all_y_pred = np.expm1(all_y_pred_log)
    
    final_r2_log = r2_score(all_y_true_log, all_y_pred_log)
    final_r2_orig = r2_score(all_y_true, all_y_pred)
    final_mae = mean_absolute_error(all_y_true, all_y_pred)
    
    print(f"   • R² Score (Log Scale): {final_r2_log:.4f} ({final_r2_log*100:.2f}%)")
    print(f"   • R² Score (Original): {final_r2_orig:.4f} ({final_r2_orig*100:.2f}%)")
    print(f"   • MAE: {final_mae:.2f} gün")
    
    # 11. Karşılaştırma
    print(f"\n📈 KARŞILAŞTIRMA:")
    print(f"   • ESKİ SİSTEM (step_16): 0.8306 (83.06%) - Log Scale")
    print(f"   • YENİ SİSTEM: {final_r2_log:.4f} ({final_r2_log*100:.2f}%) - Log Scale")
    
    improvement = final_r2_log - 0.8306
    improvement_pct = (improvement / 0.8306) * 100
    
    if improvement > 0:
        print(f"   • İYİLEŞME: +{improvement:.4f} (+{improvement_pct:.2f}%) 🚀")
    else:
        print(f"   • FARK: {improvement:.4f} ({improvement_pct:.2f}%)")
    
    print(f"\n{'='*70}")
    print(f"✅ TEST TAMAMLANDI!")
    print(f"{'='*70}")
    
    return {
        'r2_log': final_r2_log,
        'r2_orig': final_r2_orig,
        'mae': final_mae,
        'mainstream_r2': r2_main,
        'high_r2': r2_high,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }


if __name__ == "__main__":
    results = test_final_system()
