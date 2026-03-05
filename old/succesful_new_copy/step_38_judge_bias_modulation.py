"""
🚀 STEP 38 - HAKİM BIAS DÜZELTMESİ (Sabit +51 Gün)
=====================================================
Tüm tahminlere sistematik hakim etkisini yansıtmak için
sabit bir düzeltme terimi eklenir.

Model tahminleri genel olarak gerçek cezaların altında kalıyor,
çünkü hakimler geçmişte ortalama +51 gün fazladan ceza vermiş.
Bu fark istatistiksel olarak 368 hakim, binlerce dava üzerinden hesaplandı.

Yani: Nihai Tahmin = Model Çıktısı + 51 gün
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

VERI_YOLU   = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR   = Path("../model_data_v2_interactions")
JUDGE_BIAS  = 51   # Gün — 368 hakimin ortalama sertlik düzeltmesi
THRESHOLD   = 3000


def predict(case: dict) -> dict:
    """
    Dava için hakim-düzeltmeli ceza tahmini üretir.

    Parameters
    ----------
    case : Dava özelliklerini içeren dict

    Returns
    -------
    dict:
      segment          - Mainstream / High Severity
      model_prediction - Ham model çıktısı (gün)
      judge_correction - Sabit hakim düzeltmesi (gün)
      final_prediction - Nihai tahmin (gün)
    """
    KNOWN_CAT = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass']

    # ── Router ──────────────────────────────────────────────────────────────
    router = CatBoostClassifier()
    router.load_model(str(MODEL_DIR / "router_classifier_precision_with_threshold.cbm"))
    router_feats = joblib.load(MODEL_DIR / "router_features_optimized.pkl")
    router_cats  = joblib.load(MODEL_DIR / "router_cat_features_optimized.pkl")

    r_row = pd.DataFrame([{k: case.get(k, 'Unknown' if k in router_cats else 0)
                            for k in router_feats}])
    for col in r_row.columns:
        if col in router_cats or r_row[col].dtype == 'object':
            r_row[col] = r_row[col].astype(str).replace('nan', 'Unknown')
        else:
            r_row[col] = pd.to_numeric(r_row[col], errors='coerce').fillna(0)

    segment = int(router.predict(r_row)[0])

    # ── Regresyon ────────────────────────────────────────────────────────────
    if segment == 0:
        model = CatBoostRegressor()
        model.load_model(str(MODEL_DIR / "model_low_v2_optimized.cbm"))
        feats = joblib.load(MODEL_DIR / "features_v2.pkl")
    else:
        model = CatBoostRegressor()
        model.load_model(str(MODEL_DIR / "model_high_v2_comprehensive_optimized.cbm"))
        feats = joblib.load(MODEL_DIR / "features_v2_comprehensive.pkl")
        cats_high = joblib.load(MODEL_DIR / "cat_features_v2_comprehensive.pkl")
        KNOWN_CAT = cats_high

    row = pd.DataFrame([{k: case.get(k, 'Unknown' if k in KNOWN_CAT else 0)
                         for k in feats}])
    for col in row.columns:
        if col in KNOWN_CAT or row[col].dtype == 'object' or \
           row[col].dtype.name == 'category':
            row[col] = row[col].astype(str).replace('nan', 'Unknown')
        else:
            row[col] = pd.to_numeric(row[col], errors='coerce').fillna(0)

    model_pred = float(np.expm1(model.predict(row)[0]))
    final_pred = round(max(0, model_pred + JUDGE_BIAS), 1)

    return {
        'segment'          : 'Mainstream' if segment == 0 else 'High Severity',
        'model_prediction' : round(model_pred, 1),
        'judge_correction' : JUDGE_BIAS,
        'final_prediction' : final_pred,
    }


# ── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 HAKİM BIAS DÜZELTMESİ — DEMO")
    print("=" * 60)

    example = {
        'highest_severity'       : 12,
        'violent_crime'          : 1,
        'is_recid_new'           : 1,
        'year'                   : 2018,
        'wcisclass'              : 'CF',
        'release'                : 1,
        'max_hist_jail'          : 1200,
        'pct_male'               : 0.72,
        'age_judge'              : 52,
        'age_offense'            : 29,
        'pct_black'              : 0.38,
        'sex'                    : 'Male',
        'race'                   : 'African American',
        'prior_felony'           : 2,
        'prior_misdemeanor'      : 3,
        'prior_criminal_traffic' : 1,
        'avg_hist_jail'          : 700,
        'median_hist_jail'       : 650,
        'min_hist_jail'          : 350,
        'county'                 : '54',
        'case_type'              : 'CF',
        'zip'                    : '53201',
        'severity_x_violent'     : 12,
        'age_gap'                : 23,
        'violent_recid'          : 1,
        'severity_x_recid'       : 12,
    }

    result = predict(example)

    print(f"\n  Segment          : {result['segment']}")
    print(f"  Model Tahmini    : {result['model_prediction']:.0f} gün")
    print(f"  Hakim Düzeltmesi : +{result['judge_correction']} gün  (sistematik ortalama)")
    print(f"  ─────────────────────────────────────")
    print(f"  Nihai Tahmin     : {result['final_prediction']:.0f} gün  "
          f"({result['final_prediction']/365:.1f} yıl)")
    print("=" * 60)
