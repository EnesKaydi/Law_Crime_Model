
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import joblib
from pathlib import Path

# Paths
ROUTER_MODEL_PATH = Path("../model_data_router/router_classifier.cbm")
ROUTER_FEATURES_PATH = Path("../model_data_router/router_features.pkl")

LOW_MODEL_PATH = Path("../model_data_segmented/model_low_3000.cbm")
HIGH_MODEL_PATH = Path("../model_data_segmented/model_high_3000.cbm")
SEG_FEATURES_PATH = Path("../model_data_segmented/features_list.pkl")
SEG_CAT_FEATURES_PATH = Path("../model_data_segmented/cat_features_list.pkl")

def load_system():
    print("⏳ AI Yargıç Sistemi Yükleniyor...")
    
    # Router
    if not ROUTER_MODEL_PATH.exists(): raise FileNotFoundError("Router kayıp!")
    router = CatBoostClassifier()
    router.load_model(str(ROUTER_MODEL_PATH))
    router_features = joblib.load(ROUTER_FEATURES_PATH)
    
    # Uzman Modeller
    if not LOW_MODEL_PATH.exists(): raise FileNotFoundError("Düşük Ceza Modeli kayıp!")
    model_low = CatBoostRegressor()
    model_low.load_model(str(LOW_MODEL_PATH))
    
    if not HIGH_MODEL_PATH.exists(): raise FileNotFoundError("Ağır Ceza Modeli kayıp!")
    model_high = CatBoostRegressor()
    model_high.load_model(str(HIGH_MODEL_PATH))
    
    seg_features = joblib.load(SEG_FEATURES_PATH)
    seg_cat_features = joblib.load(SEG_CAT_FEATURES_PATH)
    
    print("✅ Sistem Hazır: Router + Model Low + Model High")
    return router, router_features, model_low, model_high, seg_features, seg_cat_features

def predict_case(router, router_feats, model_low, model_high, seg_feats, cat_feats, input_data):
    # Kategorik Kolonlar (Tüm modeller için ortak kabul edelim)
    # Eksik/NaN olsa bile float kalmasına izin vermemeliyiz.
    KNOWN_CAT_FEATURES = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass']
    
    # 1. Router İçin Veri Hazırla
    router_row = {}
    for f in router_feats:
        val = input_data.get(f)
        router_row[f] = val if val is not None else np.nan
            
    df_router = pd.DataFrame([router_row])
    
    # Router Kategorik Düzeltme (Zorla String)
    for col in df_router.columns:
        if col in KNOWN_CAT_FEATURES or df_router[col].dtype == 'object':
            df_router[col] = df_router[col].fillna("Unknown").astype(str)
            df_router.loc[df_router[col] == 'nan', col] = "Unknown"
            
    # 2. Router Kararı (0 veya 1)
    is_severe = router.predict(df_router)[0]
    prob_severe = router.predict_proba(df_router)[0][1]
    
    # 3. Uzman Model Seçimi ve Tahmin
    reg_row = {}
    for f in seg_feats:
        val = input_data.get(f)
        reg_row[f] = val if val is not None else np.nan
            
    df_reg = pd.DataFrame([reg_row])
    
    # Regresyon Kategorik İşlemleri (Zorunlu)
    for col in cat_feats:
        if col in df_reg.columns:
            df_reg[col] = df_reg[col].fillna("Unknown").astype(str)
            df_reg.loc[df_reg[col] == 'nan', col] = "Unknown"
            
    if is_severe == 1:
        model_name = "Ağır Ceza Mahkemesi (Model High)"
        # Model High categorical handling (safe check)
        pred_log = model_high.predict(df_reg)[0]
    else:
        model_name = "Asliye Ceza Mahkemesi (Model Low)"
        # Model Low categorical handling (safe check)
        pred_log = model_low.predict(df_reg)[0]
        
    days = np.expm1(pred_log)
    
    return {
        "predicted_days": days,
        "is_severe_prob": prob_severe,
        "routed_to": model_name
    }

# --- TEST ---
if __name__ == "__main__":
    router, r_feats, m_low, m_high, s_feats, c_feats = load_system()
    
    # Test Senaryosu: Ağır Suç (Severity 10, Recidivist)
    case_heavy = {
        'highest_severity': 10,
        'violent_crime': 1,
        'is_recid_new': 1,
        'judge_id': '673',
        'county': '40',
        'case_type': 'Felony'
    }
    
    result = predict_case(router, r_feats, m_low, m_high, s_feats, c_feats, case_heavy)
    print("\n⚖️ [AĞIR SUÇ ÖRNEĞİ]")
    print(f"Yönlendirme: {result['routed_to']} (Olasılık: %{result['is_severe_prob']*100:.1f})")
    print(f"Tahmin: {result['predicted_days']:.0f} GÜN")
    
    # Test Senaryosu: Hafif Suç (Severity 3, İlk Suç)
    case_light = {
        'highest_severity': 3,
        'violent_crime': 0,
        'is_recid_new': 0,
        'judge_id': '221',
        'county': '13',
        'case_type': 'Misdemeanor'
    }
    
    result2 = predict_case(router, r_feats, m_low, m_high, s_feats, c_feats, case_light)
    print("\n⚖️ [HAFİF SUÇ ÖRNEĞİ]")
    print(f"Yönlendirme: {result2['routed_to']} (Olasılık: %{result2['is_severe_prob']*100:.1f})")
    print(f"Tahmin: {result2['predicted_days']:.0f} GÜN")
