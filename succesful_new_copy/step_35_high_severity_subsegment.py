"""
🚀 STEP 35 - HIGH SEVERITY SUB-SEGMENTATION
=============================================
Ağır suçları (>3000 gün) kendi içinde 2 alt segmente böl:
  - Orta-Ağır: 3000-6000 gün arası
  - Çok-Ağır:  6000+ gün

Her birine ayrı uzman model kur ve skorları karşılaştır.
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
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# Alt Segment Sınırları
LOW_THRESHOLD  = 3000   # Mainstream / High sınırı
MID_THRESHOLD  = 6000   # Orta-Ağır / Çok-Ağır sınırı


def create_comprehensive_features(df):
    df = df.copy()

    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
        df['age_ratio'] = age_j / (age_o + 1)
        df['age_product'] = age_j * age_o
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
    if 'highest_severity' in df.columns and 'is_recid_new' in df.columns:
        df['severity_x_recid'] = df['highest_severity'] * df['is_recid_new']

    if 'judge_id' in df.columns:
        df['judge_mean_sentence']   = df.groupby('judge_id')['jail'].transform('mean')
        df['judge_median_sentence'] = df.groupby('judge_id')['jail'].transform('median')
        df['judge_std_sentence']    = df.groupby('judge_id')['jail'].transform('std').fillna(0)
        df['judge_case_count']      = df.groupby('judge_id')['jail'].transform('count')
    if 'county' in df.columns:
        df['county_mean_sentence']   = df.groupby('county')['jail'].transform('mean')
        df['county_median_sentence'] = df.groupby('county')['jail'].transform('median')
    if 'wcisclass' in df.columns:
        df['wcisclass_mean_sentence']   = df.groupby('wcisclass')['jail'].transform('mean')
        df['wcisclass_median_sentence'] = df.groupby('wcisclass')['jail'].transform('median')
    if 'case_type' in df.columns:
        df['case_type_mean_sentence'] = df.groupby('case_type')['jail'].transform('mean')

    if 'judge_id' in df.columns and 'wcisclass' in df.columns:
        df['judge_crime_combo'] = df['judge_id'].astype(str) + '_' + df['wcisclass'].astype(str)
        df['judge_crime_mean']  = df.groupby('judge_crime_combo')['jail'].transform('mean')
    if 'judge_id' in df.columns and 'county' in df.columns:
        df['judge_county_combo'] = df['judge_id'].astype(str) + '_' + df['county'].astype(str)
        df['judge_county_mean']  = df.groupby('judge_county_combo')['jail'].transform('mean')

    prior_cols = ['prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic']
    avail = [c for c in prior_cols if c in df.columns]
    if avail:
        df['total_prior'] = df[avail].fillna(0).sum(axis=1)
        if 'violent_crime' in df.columns:
            df['violent_x_prior'] = df['violent_crime'] * df['total_prior']
        if 'highest_severity' in df.columns:
            df['severity_x_total_prior'] = df['highest_severity'] * df['total_prior']
            df['severity_per_prior']     = df['highest_severity'] / (df['total_prior'] + 1)

    if 'age_offense' in df.columns:
        df['age_bin'] = pd.cut(df['age_offense'].fillna(df['age_offense'].mean()),
                               bins=[0,25,35,45,55,100],
                               labels=['very_young','young','middle','mature','senior'])
    if 'highest_severity' in df.columns:
        df['severity_bin'] = pd.cut(df['highest_severity'],
                                    bins=[0,5,10,15,20],
                                    labels=['low','medium','high','very_high'])
        df['severity_squared'] = df['highest_severity'] ** 2
        df['severity_cubed']   = df['highest_severity'] ** 3

    if 'age_offense' in df.columns:
        af = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_squared'] = af ** 2

    if 'year' in df.columns:
        df['years_since_2000'] = df['year'] - 2000
        df['decade']           = (df['year'] // 10) * 10

    risk = []
    if 'violent_crime' in df.columns:  risk.append(df['violent_crime'] * 3)
    if 'is_recid_new' in df.columns:   risk.append(df['is_recid_new']  * 2)
    if 'highest_severity' in df.columns: risk.append(df['highest_severity'] / 10)
    if risk:
        df['composite_risk_score'] = sum(risk)

    return df


def train_and_eval_segment(df_seg, seg_name, cat_features_ref, features_ref, model_save_name):
    """Verilen alt segment için model eğit ve metrikleri döndür."""
    df_seg = df_seg.copy()
    df_seg = create_comprehensive_features(df_seg)

    # Mevcut comprehensive feature listesini kullan
    avail_features = [f for f in features_ref if f in df_seg.columns]
    X = df_seg[avail_features].copy()
    y = np.log1p(df_seg['jail'])

    # Kategorik işlemler
    cat_feats = [c for c in cat_features_ref if c in X.columns]
    for col in X.columns:
        if col in cat_feats or X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype(str).replace('nan','Unknown').replace('None','Unknown')
        else:
            X[col] = X[col].fillna(X[col].mean())

    if len(X) < 100:
        print(f"  ⚠️  {seg_name}: Yeterli veri yok ({len(X)} satır), atlanıyor.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = CatBoostRegressor(
        iterations=700,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=0.5,
        random_seed=RANDOM_STATE,
        verbose=0,
        eval_metric='RMSE',
        cat_features=cat_feats,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    preds_log  = model.predict(X_test)
    preds_orig = np.expm1(preds_log)
    y_test_orig = np.expm1(y_test)

    r2  = r2_score(y_test, preds_log)
    mae = mean_absolute_error(y_test_orig, preds_orig)

    print(f"\n  📊 {seg_name}:")
    print(f"     Vaka Sayısı : {len(df_seg):,}")
    print(f"     R² (Log)   : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"     MAE        : {mae:.1f} gün")

    model.save_model(str(MODEL_DIR / model_save_name))
    return {'r2': r2, 'mae': mae, 'n': len(df_seg)}


def run():
    print("=" * 70)
    print("🚀 STEP 35 - HIGH SEVERITY SUB-SEGMENTATION")
    print("=" * 70)

    df = pd.read_csv(VERI_YOLU, low_memory=False)
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()

    df_high = df[df['jail'] > LOW_THRESHOLD].copy()
    print(f"\n📌 Toplam Ağır Suç Vakası      : {len(df_high):,}  (3000+ gün)")

    df_mid  = df_high[df_high['jail'] <= MID_THRESHOLD].copy()
    df_very = df_high[df_high['jail'] >  MID_THRESHOLD].copy()

    print(f"   → Orta-Ağır (3000-6000 gün) : {len(df_mid):,}")
    print(f"   → Çok-Ağır  (6000+  gün)    : {len(df_very):,}")

    # Mevcut feature ve cat listelerini yükle
    features_ref     = joblib.load(MODEL_DIR / "features_v2_comprehensive.pkl")
    cat_features_ref = joblib.load(MODEL_DIR / "cat_features_v2_comprehensive.pkl")

    print("\n🔹 Orta-Ağır Model Eğitiliyor (3000-6000 gün)...")
    res_mid = train_and_eval_segment(
        df_mid, "Orta-Ağır (3000-6000)",
        cat_features_ref, features_ref,
        "model_mid_severity.cbm"
    )

    print("\n🔹 Çok-Ağır Model Eğitiliyor (6000+ gün)...")
    res_very = train_and_eval_segment(
        df_very, "Çok-Ağır (6000+)",
        cat_features_ref, features_ref,
        "model_very_high_severity.cbm"
    )

    print("\n" + "=" * 70)
    print("📈 SONUÇ KARŞILAŞTIRMASI")
    print("=" * 70)
    print(f"  Eski Birleşik Model (3000+) : R² ≈ %61.17")
    if res_mid:
        print(f"  Yeni Orta-Ağır  (3000-6000): R² = {res_mid['r2']*100:.2f}%  MAE={res_mid['mae']:.0f} gün")
    if res_very:
        print(f"  Yeni Çok-Ağır   (6000+)    : R² = {res_very['r2']*100:.2f}%  MAE={res_very['mae']:.0f} gün")
    print("=" * 70)


if __name__ == "__main__":
    run()
