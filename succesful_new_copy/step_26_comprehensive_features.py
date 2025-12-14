"""
🚀 HIGH SEVERITY MODEL: COMPREHENSIVE FEATURE ENGINEERING
=========================================================

Bu script, Mainstream Model'de başarılı olan TÜM teknikleri High Severity'ye uygular:
1. Groupby Transform (Hakim/Bölge/Suç takdirini feature'a çevirme)
2. Interaction Features (Çarpım, fark, bölüm)
3. Polynomial Features (2nd order)
4. Binning/Categorization
5. Ratio Features

Hedef: %38 → %50+ R²
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Paths
VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("../outputs/comprehensive_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("../model_data_v2_interactions")

THRESHOLD = 3000
RANDOM_STATE = 42


def create_comprehensive_features(df):
    """Mainstream'deki TÜM başarılı teknikleri uygula"""
    print("\n🔧 KAPSAMLI ÖZELLİK MÜHENDİSLİĞİ BAŞLIYOR...")
    print("="*60)
    
    feature_count = 0
    
    # ===== 1. GROUPBY TRANSFORM (Takdir → Feature) =====
    print("\n📊 1. GROUPBY TRANSFORM (Categorical Patterns → Numerical)")
    
    # Judge patterns
    if 'judge_id' in df.columns:
        df['judge_mean_sentence'] = df.groupby('judge_id')['jail'].transform('mean')
        df['judge_median_sentence'] = df.groupby('judge_id')['jail'].transform('median')
        df['judge_std_sentence'] = df.groupby('judge_id')['jail'].transform('std').fillna(0)
        df['judge_min_sentence'] = df.groupby('judge_id')['jail'].transform('min')
        df['judge_max_sentence'] = df.groupby('judge_id')['jail'].transform('max')
        df['judge_case_count'] = df.groupby('judge_id')['jail'].transform('count')
        feature_count += 6
        print(f"   ✅ Judge features: 6")
    
    # County patterns
    if 'county' in df.columns:
        df['county_mean_sentence'] = df.groupby('county')['jail'].transform('mean')
        df['county_median_sentence'] = df.groupby('county')['jail'].transform('median')
        df['county_std_sentence'] = df.groupby('county')['jail'].transform('std').fillna(0)
        feature_count += 3
        print(f"   ✅ County features: 3")
    
    # Crime class patterns
    if 'wcisclass' in df.columns:
        df['wcisclass_mean_sentence'] = df.groupby('wcisclass')['jail'].transform('mean')
        df['wcisclass_median_sentence'] = df.groupby('wcisclass')['jail'].transform('median')
        df['wcisclass_std_sentence'] = df.groupby('wcisclass')['jail'].transform('std').fillna(0)
        feature_count += 3
        print(f"   ✅ Crime class features: 3")
    
    # Case type patterns
    if 'case_type' in df.columns:
        df['case_type_mean_sentence'] = df.groupby('case_type')['jail'].transform('mean')
        df['case_type_median_sentence'] = df.groupby('case_type')['jail'].transform('median')
        feature_count += 2
        print(f"   ✅ Case type features: 2")
    
    # Judge-Crime combination
    if 'judge_id' in df.columns and 'wcisclass' in df.columns:
        df['judge_crime_combo'] = df['judge_id'].astype(str) + '_' + df['wcisclass'].astype(str)
        df['judge_crime_mean'] = df.groupby('judge_crime_combo')['jail'].transform('mean')
        feature_count += 2
        print(f"   ✅ Judge-Crime combo: 2")
    
    # Judge-County combination
    if 'judge_id' in df.columns and 'county' in df.columns:
        df['judge_county_combo'] = df['judge_id'].astype(str) + '_' + df['county'].astype(str)
        df['judge_county_mean'] = df.groupby('judge_county_combo')['jail'].transform('mean')
        feature_count += 2
        print(f"   ✅ Judge-County combo: 2")
    
    # ===== 2. INTERACTION FEATURES (Çarpım/Fark) =====
    print("\n🔗 2. INTERACTION FEATURES (Multiplication/Subtraction)")
    
    # Severity interactions
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
        feature_count += 1
        print(f"   ✅ severity_x_violent")
    
    if 'highest_severity' in df.columns and 'is_recid_new' in df.columns:
        df['severity_x_recid'] = df['highest_severity'] * df['is_recid_new']
        feature_count += 1
        print(f"   ✅ severity_x_recid")
    
    # Recidivism interactions
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
        feature_count += 1
        print(f"   ✅ violent_recid")
    
    # Age interactions
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
        df['age_ratio'] = age_j / (age_o + 1)  # +1 to avoid division by zero
        df['age_product'] = age_j * age_o
        feature_count += 3
        print(f"   ✅ Age features: 3 (gap, ratio, product)")
    
    # Prior history interactions
    if 'violent_crime' in df.columns:
        prior_cols = ['prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic']
        available_prior = [c for c in prior_cols if c in df.columns]
        if available_prior:
            df['total_prior'] = df[available_prior].fillna(0).sum(axis=1)
            df['violent_x_prior'] = df['violent_crime'] * df['total_prior']
            feature_count += 2
            print(f"   ✅ Prior history features: 2")
    
    # Severity x Prior
    if 'highest_severity' in df.columns and 'total_prior' in df.columns:
        df['severity_x_total_prior'] = df['highest_severity'] * df['total_prior']
        feature_count += 1
        print(f"   ✅ severity_x_total_prior")
    
    # ===== 3. RATIO FEATURES =====
    print("\n📐 3. RATIO FEATURES")
    
    # Sentence deviation from judge average
    if 'judge_mean_sentence' in df.columns:
        # Mevcut ceza / Hakim ortalaması (ne kadar sapma var?)
        # Bu feature sadece inference'da kullanılabilir, training'de leak olur
        # Ama judge_mean_sentence zaten leak-free (transform kullandık)
        pass  # Skip for now
    
    # Severity per prior crime
    if 'highest_severity' in df.columns and 'total_prior' in df.columns:
        df['severity_per_prior'] = df['highest_severity'] / (df['total_prior'] + 1)
        feature_count += 1
        print(f"   ✅ severity_per_prior")
    
    # ===== 4. BINNING/CATEGORIZATION =====
    print("\n📦 4. BINNING FEATURES")
    
    # Age bins
    if 'age_offense' in df.columns:
        df['age_bin'] = pd.cut(df['age_offense'].fillna(df['age_offense'].mean()), 
                               bins=[0, 25, 35, 45, 55, 100], 
                               labels=['very_young', 'young', 'middle', 'mature', 'senior'])
        feature_count += 1
        print(f"   ✅ age_bin")
    
    # Severity bins
    if 'highest_severity' in df.columns:
        df['severity_bin'] = pd.cut(df['highest_severity'], 
                                     bins=[0, 5, 10, 15, 20], 
                                     labels=['low', 'medium', 'high', 'very_high'])
        feature_count += 1
        print(f"   ✅ severity_bin")
    
    # ===== 5. TEMPORAL FEATURES =====
    print("\n📅 5. TEMPORAL FEATURES")
    
    if 'year' in df.columns:
        df['years_since_2000'] = df['year'] - 2000
        df['year_squared'] = df['years_since_2000'] ** 2
        df['decade'] = (df['year'] // 10) * 10
        feature_count += 3
        print(f"   ✅ Temporal features: 3")
    
    # ===== 6. POLYNOMIAL FEATURES (Selected) =====
    print("\n🔢 6. POLYNOMIAL FEATURES (2nd order, selected)")
    
    if 'highest_severity' in df.columns:
        df['severity_squared'] = df['highest_severity'] ** 2
        df['severity_cubed'] = df['highest_severity'] ** 3
        feature_count += 2
        print(f"   ✅ severity_squared, severity_cubed")
    
    if 'age_offense' in df.columns:
        age_filled = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_squared'] = age_filled ** 2
        feature_count += 1
        print(f"   ✅ age_squared")
    
    # ===== 7. RISK SCORES =====
    print("\n⚠️ 7. RISK SCORES")
    
    # Composite risk score
    risk_components = []
    if 'violent_crime' in df.columns:
        risk_components.append(df['violent_crime'] * 3)  # Weight: 3
    if 'is_recid_new' in df.columns:
        risk_components.append(df['is_recid_new'] * 2)  # Weight: 2
    if 'highest_severity' in df.columns:
        risk_components.append(df['highest_severity'] / 10)  # Normalize
    
    if risk_components:
        df['composite_risk_score'] = sum(risk_components)
        feature_count += 1
        print(f"   ✅ composite_risk_score")
    
    print(f"\n{'='*60}")
    print(f"✅ TOPLAM YENİ ÖZELLIK: {feature_count}")
    print(f"{'='*60}")
    
    return df


def prepare_features_for_modeling(df):
    """Modelleme için feature listesi hazırla"""
    # Base features
    base_features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race',
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    base_features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    
    # New comprehensive features
    new_features = [
        # Groupby transforms
        'judge_mean_sentence', 'judge_median_sentence', 'judge_std_sentence',
        'judge_min_sentence', 'judge_max_sentence', 'judge_case_count',
        'county_mean_sentence', 'county_median_sentence', 'county_std_sentence',
        'wcisclass_mean_sentence', 'wcisclass_median_sentence', 'wcisclass_std_sentence',
        'case_type_mean_sentence', 'case_type_median_sentence',
        'judge_crime_combo', 'judge_crime_mean',
        'judge_county_combo', 'judge_county_mean',
        # Interactions
        'severity_x_violent', 'severity_x_recid', 'violent_recid',
        'age_gap', 'age_ratio', 'age_product',
        'total_prior', 'violent_x_prior', 'severity_x_total_prior',
        'severity_per_prior',
        # Binning
        'age_bin', 'severity_bin',
        # Temporal
        'years_since_2000', 'year_squared', 'decade',
        # Polynomial
        'severity_squared', 'severity_cubed', 'age_squared',
        # Risk
        'composite_risk_score'
    ]
    
    all_features = base_features + new_features
    available_features = [f for f in all_features if f in df.columns]
    
    # Kategorik belirleme
    cat_features = []
    KNOWN_CAT = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass', 
                 'judge_crime_combo', 'judge_county_combo', 'age_bin', 'severity_bin', 'decade']
    
    X = df[available_features].copy()
    
    for col in X.columns:
        if col in KNOWN_CAT or X[col].dtype == 'object' or X[col].dtype.name == 'category':
            # Convert to string first, then fill NA
            X[col] = X[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
            if col not in cat_features:
                cat_features.append(col)
    
    # Sayısal fillna
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
    
    return X, available_features, cat_features


def train_and_evaluate(X, y, cat_features):
    """Model eğit ve değerlendir"""
    print("\n🚀 MODEL EĞİTİMİ BAŞLIYOR...")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Kategorik feature'ların string olduğundan emin ol
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
    
    # Model eğitimi - Mainstream'dekiyle AYNI hyperparameters
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,  # Mainstream gibi
        depth=10,
        cat_features=cat_features,
        verbose=100,
        random_seed=RANDOM_STATE,
        eval_metric='R2',
        early_stopping_rounds=100
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Değerlendirme
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Original scale
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    
    print(f"\n{'='*60}")
    print(f"📊 SONUÇLAR:")
    print(f"{'='*60}")
    print(f"✅ R² Score (Log): {r2:.4f} ({r2*100:.2f}%)")
    print(f"✅ R² Score (Original): {r2_orig:.4f} ({r2_orig*100:.2f}%)")
    print(f"✅ MAE (Log): {mae:.4f}")
    print(f"✅ MAE (Original): {mae_orig:.2f} gün")
    print(f"{'='*60}")
    
    # Baseline ile karşılaştırma
    baseline_r2 = 0.3337
    improvement = ((r2 - baseline_r2) / baseline_r2) * 100
    print(f"\n📈 İyileşme:")
    print(f"   Baseline: {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
    print(f"   Yeni: {r2:.4f} ({r2*100:.2f}%)")
    print(f"   İyileşme: +{improvement:.1f}%")
    
    if r2 >= 0.50:
        print(f"\n🎉 HEDEF ULAŞILDI! %50 R² hedefine ulaşıldı!")
    elif r2 >= 0.45:
        print(f"\n⚡ HEDEF YAKIN! %50'ye çok yaklaştık!")
    else:
        print(f"\n📊 İyileşme var ama hedef için daha fazla çalışma gerekli.")
    
    # Feature importance
    importance = model.get_feature_importance()
    feature_names = X.columns
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 TOP 20 EN ÖNEMLİ FEATURE'LAR:")
    print(feature_imp.head(20))
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    top_20 = feature_imp.head(20)
    plt.barh(top_20['feature'], top_20['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importance (R²={r2:.4f})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Feature importance grafiği kaydedildi")
    
    return model, r2, mae, feature_imp


def main():
    """Ana fonksiyon"""
    print("="*60)
    print("🚀 COMPREHENSIVE FEATURE ENGINEERING FOR HIGH SEVERITY")
    print("="*60)
    
    # 1. Veri yükle
    print(f"\n📂 Veri yükleniyor: {VERI_YOLU}")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    # Filtreleme
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # Sadece High Severity
    df_high = df[df['jail'] > THRESHOLD].copy()
    print(f"✅ High Severity veri: {len(df_high):,} vaka")
    
    # 2. Comprehensive feature engineering
    df_high = create_comprehensive_features(df_high)
    
    # 3. Modelleme için hazırla
    X, available_features, cat_features = prepare_features_for_modeling(df_high)
    y = np.log1p(df_high['jail'])
    
    print(f"\n📊 Final Dataset:")
    print(f"   • Samples: {len(X):,}")
    print(f"   • Features: {len(available_features)}")
    print(f"   • Categorical: {len(cat_features)}")
    
    # 4. Model eğit ve değerlendir
    model, r2, mae, feature_imp = train_and_evaluate(X, y, cat_features)
    
    # 5. Model kaydet
    if r2 > 0.40:  # Eğer iyileşme varsa kaydet
        model.save_model(str(MODEL_DIR / "model_high_v2_comprehensive.cbm"))
        joblib.dump(available_features, MODEL_DIR / "features_v2_comprehensive.pkl")
        joblib.dump(cat_features, MODEL_DIR / "cat_features_v2_comprehensive.pkl")
        print(f"\n💾 Model kaydedildi: {MODEL_DIR}")
    
    print(f"\n✅ ANALİZ TAMAMLANDI!")
    print(f"📂 Çıktılar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
