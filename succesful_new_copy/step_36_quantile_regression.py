"""
🚀 STEP 36 - QUANTILE REGRESSION FOR HIGH SEVERITY
====================================================
CatBoost'un farklı loss fonksiyonlarını dene:
  - MAPE     : Yüzdesel hata odaklı (büyük değerlerde daha adil)
  - Quantile : Medyan tahmini (outlier'lara daha dayanıklı)
  - MAE      : Mutlak hata odaklı (RMSE'nin aksine uç değere duyarlı değil)

Her loss ile aynı veriyi eğit, R² ve MAE karşılaştır.
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
RANDOM_STATE = 42
THRESHOLD = 3000


def create_comprehensive_features(df):
    df = df.copy()
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
        df['age_ratio'] = age_j / (age_o + 1)
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
        df['severity_bin']     = pd.cut(df['highest_severity'], bins=[0,5,10,15,20],
                                         labels=['low','medium','high','very_high'])
        df['severity_squared'] = df['highest_severity'] ** 2
        df['severity_cubed']   = df['highest_severity'] ** 3
    if 'age_offense' in df.columns:
        df['age_squared'] = df['age_offense'].fillna(df['age_offense'].mean()) ** 2
    if 'year' in df.columns:
        df['years_since_2000'] = df['year'] - 2000
        df['decade']           = (df['year'] // 10) * 10
    risk = []
    if 'violent_crime' in df.columns:    risk.append(df['violent_crime'] * 3)
    if 'is_recid_new' in df.columns:     risk.append(df['is_recid_new']  * 2)
    if 'highest_severity' in df.columns: risk.append(df['highest_severity'] / 10)
    if risk:
        df['composite_risk_score'] = sum(risk)
    return df


def run():
    print("=" * 70)
    print("🚀 STEP 36 - QUANTILE REGRESSION COMPARISON (HIGH SEVERITY)")
    print("=" * 70)

    df = pd.read_csv(VERI_YOLU, low_memory=False)
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()

    df_high = df[df['jail'] > THRESHOLD].copy()
    print(f"\n📌 High Severity Vaka Sayısı: {len(df_high):,}  (3000+ gün)")

    df_high = create_comprehensive_features(df_high)

    features_ref     = joblib.load(MODEL_DIR / "features_v2_comprehensive.pkl")
    cat_features_ref = joblib.load(MODEL_DIR / "cat_features_v2_comprehensive.pkl")
    avail_features   = [f for f in features_ref if f in df_high.columns]

    X = df_high[avail_features].copy()
    y_log  = np.log1p(df_high['jail'])
    y_orig = df_high['jail'].values

    cat_feats = [c for c in cat_features_ref if c in X.columns]
    for col in X.columns:
        if col in cat_feats or X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype(str).replace('nan','Unknown').replace('None','Unknown')
        else:
            X[col] = X[col].fillna(X[col].mean())

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE)
    # Orijinal scale test için aynı index'ten y_orig kullan
    _, _, y_train_o, y_test_o = train_test_split(
        X, y_orig, test_size=0.2, random_state=RANDOM_STATE)

    # --- Denenecek Loss Fonksiyonları ---
    experiments = [
        # (İsim, loss_function, ek_params)
        ("RMSE (Mevcut Baz)",    "RMSE",            {}),
        ("MAE (Sağlam Hata)",    "MAE",             {}),
        ("MAPE (Yüzdesel Hata)", "MAPE",            {}),
        ("Quantile 0.5 (Medyan)","Quantile:alpha=0.5", {}),
    ]

    results = []
    best_model = None
    best_r2 = -np.inf

    for name, loss_fn, extra in experiments:
        print(f"\n⚗️  Deney: {name}  (loss={loss_fn})")
        try:
            model = CatBoostRegressor(
                iterations=600,
                learning_rate=0.06,
                depth=8,
                l2_leaf_reg=1.0,
                loss_function=loss_fn,
                random_seed=RANDOM_STATE,
                verbose=0,
                cat_features=cat_feats,
                early_stopping_rounds=40,
                **extra
            )
            model.fit(X_train_l, y_train_l, eval_set=(X_test_l, y_test_l))

            preds_log  = model.predict(X_test_l)
            preds_orig = np.expm1(preds_log)

            r2  = r2_score(y_test_l, preds_log)
            mae = mean_absolute_error(np.expm1(y_test_l), preds_orig)

            print(f"   → R² (Log): {r2*100:.2f}%   MAE: {mae:.0f} gün")
            results.append({'name': name, 'r2': r2, 'mae': mae})

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name  = name

        except Exception as e:
            print(f"   ❌ Hata: {e}")

    print("\n" + "=" * 70)
    print("📈 SONUÇ TABLOSU")
    print("=" * 70)
    print(f"{'Model':<30} {'R²':>8}  {'MAE':>10}")
    print("-" * 52)
    for r in results:
        print(f"  {r['name']:<28} {r['r2']*100:>7.2f}%  {r['mae']:>8.0f} gün")
    print("=" * 70)
    print(f"\n🏆 En İyi Loss Fonksiyonu: {best_name}  (R²={best_r2*100:.2f}%)")

    if best_model:
        best_model.save_model(str(MODEL_DIR / "model_high_v2_best_loss.cbm"))
        print(f"💾 En İyi Model Kaydedildi: model_high_v2_best_loss.cbm")


if __name__ == "__main__":
    run()
