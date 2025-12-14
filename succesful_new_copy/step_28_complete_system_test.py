"""
📊 COMPLETE SYSTEM PERFORMANCE TEST
====================================

Tüm sistemi test et:
1. Router ile doğru modele yönlendir
2. Her model kendi segmentinde tahmin yapsın
3. Genel R² hesapla (Original + Log scale)

Bu, step_16'daki gibi GERÇEK performans testi!
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

# Paths
VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("../model_data_v2_interactions")

THRESHOLD = 3000
RANDOM_STATE = 42


def create_interactions(df):
    """Interaction features oluştur"""
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
    
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
    
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
    
    return df


def test_complete_system():
    """Tüm sistemi test et - step_16 gibi"""
    print("="*70)
    print("🚀 COMPLETE SYSTEM PERFORMANCE TEST")
    print("="*70)
    
    # 1. Veri yükle
    print(f"\n📂 Veri yükleniyor...")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    # Filtreleme
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    print(f"✅ Toplam vaka: {len(df):,}")
    
    # Interaction features ekle
    df = create_interactions(df)
    
    # Feature listesi
    base_features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race',
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    base_features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    new_features = ['severity_x_violent', 'age_gap', 'violent_recid']
    all_features = base_features + new_features
    final_features = [f for f in all_features if f in df.columns]
    
    # Kategorik belirleme
    cat_features = []
    KNOWN_CAT_FEATURES = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass']
    
    X = df[final_features].copy()
    
    for col in X.columns:
        if col in KNOWN_CAT_FEATURES or X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("Unknown").astype(str)
            if col not in cat_features:
                cat_features.append(col)
    
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
    
    print(f"📌 Features: {len(final_features)}")
    print(f"📌 Categorical: {len(cat_features)}")
    
    # 2. Modelleri yükle
    print(f"\n📦 Modeller yükleniyor...")
    
    # Router
    router = CatBoostClassifier()
    router.load_model(str(MODEL_DIR / "router_v2.cbm"))
    print(f"   ✅ Router yüklendi")
    
    # Mainstream Model
    model_mainstream = CatBoostRegressor()
    model_mainstream.load_model(str(MODEL_DIR / "model_low_v2.cbm"))
    print(f"   ✅ Mainstream Model yüklendi")
    
    # High Severity Model - ESKİ modeli kullan (41 feature)
    model_high = CatBoostRegressor()
    model_high.load_model(str(MODEL_DIR / "model_high_v2.cbm"))
    print(f"   ✅ High Severity Model (V2 - 41 features) yüklendi")
    using_comprehensive = False
    
    print(f"\n   ⚠️ NOT: Comprehensive model (75 features) test için uygun değil")
    print(f"   → Şimdi eski modelle (41 features) genel sistem performansını test ediyoruz")
    print(f"   → Comprehensive model sadece High Severity segment için eğitildi")
    
    # 3. Test seti oluştur
    print(f"\n🧪 Test seti hazırlanıyor...")
    
    # Tüm veriyi train/test'e ayır
    y_router = (df['jail'] > THRESHOLD).astype(int)
    X_train, X_test, y_train_router, y_test_router = train_test_split(
        X, y_router, test_size=0.2, random_state=RANDOM_STATE, stratify=y_router
    )
    
    # Test setindeki gerçek jail değerleri
    test_indices = X_test.index
    y_test_jail = df.loc[test_indices, 'jail'].values
    y_test_log = np.log1p(y_test_jail)
    
    print(f"   • Test seti: {len(X_test):,} vaka")
    print(f"   • Mainstream: {(y_test_router == 0).sum():,} vaka")
    print(f"   • High Severity: {(y_test_router == 1).sum():,} vaka")
    
    # 4. Router ile tahmin
    print(f"\n🎯 Router ile yönlendirme...")
    router_predictions = router.predict(X_test)
    router_accuracy = (router_predictions == y_test_router).mean()
    
    print(f"   ✅ Router Accuracy: {router_accuracy:.4f} ({router_accuracy*100:.2f}%)")
    
    # 5. Her segment için tahmin
    print(f"\n🔮 Segment bazlı tahminler...")
    
    predictions_log = np.zeros(len(X_test))
    
    # Mainstream predictions
    mainstream_mask = (router_predictions == 0)
    if mainstream_mask.sum() > 0:
        predictions_log[mainstream_mask] = model_mainstream.predict(X_test[mainstream_mask])
        print(f"   ✅ Mainstream: {mainstream_mask.sum():,} tahmin")
    
    # High Severity predictions
    high_mask = (router_predictions == 1)
    if high_mask.sum() > 0:
        predictions_log[high_mask] = model_high.predict(X_test[high_mask])
        print(f"   ✅ High Severity: {high_mask.sum():,} tahmin")
    
    # 6. Performans hesaplama
    print(f"\n📊 PERFORMANS HESAPLAMA:")
    print("="*70)
    
    # Log scale
    r2_log = r2_score(y_test_log, predictions_log)
    mae_log = mean_absolute_error(y_test_log, predictions_log)
    
    # Original scale
    predictions_orig = np.expm1(predictions_log)
    r2_orig = r2_score(y_test_jail, predictions_orig)
    mae_orig = mean_absolute_error(y_test_jail, predictions_orig)
    
    print(f"\n🏆 GENEL SİSTEM PERFORMANSI:")
    print(f"   • R² Score (Log Scale): {r2_log:.4f} ({r2_log*100:.2f}%)")
    print(f"   • R² Score (Original Scale): {r2_orig:.4f} ({r2_orig*100:.2f}%)")
    print(f"   • MAE (Log): {mae_log:.4f}")
    print(f"   • MAE (Original): {mae_orig:.2f} gün")
    
    # Segment bazlı performans
    print(f"\n📊 SEGMENT BAZLI PERFORMANS:")
    
    # Mainstream segment
    mainstream_true_mask = (y_test_router == 0)
    if mainstream_true_mask.sum() > 0:
        mainstream_r2 = r2_score(
            y_test_log[mainstream_true_mask],
            predictions_log[mainstream_true_mask]
        )
        print(f"   • Mainstream R²: {mainstream_r2:.4f} ({mainstream_r2*100:.2f}%)")
    
    # High Severity segment
    high_true_mask = (y_test_router == 1)
    if high_true_mask.sum() > 0:
        high_r2 = r2_score(
            y_test_log[high_true_mask],
            predictions_log[high_true_mask]
        )
        print(f"   • High Severity R²: {high_r2:.4f} ({high_r2*100:.2f}%)")
    
    # Karşılaştırma
    print(f"\n📈 KARŞILAŞTIRMA:")
    print(f"   • Eski Sistem (step_16): ~0.8306 (83.06%) - Log Scale")
    print(f"   • Yeni Sistem: {r2_log:.4f} ({r2_log*100:.2f}%) - Log Scale")
    
    if using_comprehensive:
        print(f"\n✅ COMPREHENSIVE MODEL KULLANILDI!")
    else:
        print(f"\n⚠️ ESKİ MODEL KULLANILDI (Comprehensive model bulunamadı)")
    
    print(f"\n{'='*70}")
    print(f"✅ TEST TAMAMLANDI!")
    print(f"{'='*70}")
    
    return {
        'r2_log': r2_log,
        'r2_orig': r2_orig,
        'mae_log': mae_log,
        'mae_orig': mae_orig,
        'router_accuracy': router_accuracy,
        'mainstream_r2': mainstream_r2 if mainstream_true_mask.sum() > 0 else None,
        'high_r2': high_r2 if high_true_mask.sum() > 0 else None,
        'using_comprehensive': using_comprehensive
    }


if __name__ == "__main__":
    results = test_complete_system()
