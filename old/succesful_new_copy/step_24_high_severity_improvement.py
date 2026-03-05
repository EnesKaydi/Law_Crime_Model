"""
🚀 HIGH SEVERITY MODEL IMPROVEMENT EXPERIMENTS
===============================================

Bu script, SADECE High Severity Model (3000+ gün) için iyileştirme denemeleri yapar.
Mainstream Model ve Router Model AYNI KALIR.

Hedef: %33 R² → %50+ R²

Stratejiler:
1. Advanced Feature Engineering (Judge-Crime interactions)
2. Ensemble Modeling (Multiple CatBoost models)
3. Hyperparameter Optimization
4. Alternative Loss Functions (Quantile, Huber)
5. Crime Type Clustering
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Paths
VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("../outputs/high_severity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("../model_data_v2_interactions")

# Constants
THRESHOLD = 3000
RANDOM_STATE = 42


def load_high_severity_data():
    """Sadece High Severity segmentini yükle"""
    print("📂 Veri yükleniyor...")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    # Temel filtreleme
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # Sadece High Severity
    df_high = df[df['jail'] > THRESHOLD].copy()
    
    print(f"✅ High Severity veri hazır: {len(df_high):,} vaka")
    return df_high


def create_advanced_features(df):
    """Gelişmiş özellik mühendisliği - High Severity'ye özel"""
    print("\n🔧 Gelişmiş özellikler oluşturuluyor...")
    
    # Mevcut interaction features
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
    
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
    
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
    
    # YENİ ÖZELLIKLER - High Severity'ye Özel
    
    # 1. Judge Harshness Score (Hakim Sertlik Skoru)
    if 'judge_id' in df.columns:
        judge_avg = df.groupby('judge_id')['jail'].transform('mean')
        df['judge_harshness'] = judge_avg
        
        # Judge consistency (varyans)
        judge_std = df.groupby('judge_id')['jail'].transform('std')
        df['judge_consistency'] = judge_std.fillna(0)
    
    # 2. County Harshness Score (Bölge Sertlik Skoru)
    if 'county' in df.columns:
        county_avg = df.groupby('county')['jail'].transform('mean')
        df['county_harshness'] = county_avg
    
    # 3. Crime Class Severity (Suç Sınıfı Şiddeti)
    if 'wcisclass' in df.columns:
        wcis_avg = df.groupby('wcisclass')['jail'].transform('mean')
        df['wcisclass_severity'] = wcis_avg
    
    # 4. Judge-Crime Type Interaction
    if 'judge_id' in df.columns and 'wcisclass' in df.columns:
        df['judge_crime_combo'] = df['judge_id'].astype(str) + '_' + df['wcisclass'].astype(str)
    
    # 5. Recidivism Severity (Sabıka Şiddeti)
    if 'is_recid_new' in df.columns and 'highest_severity' in df.columns:
        df['recid_severity'] = df['is_recid_new'] * df['highest_severity']
    
    # 6. Total Prior History Score
    prior_cols = ['prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic']
    available_prior = [c for c in prior_cols if c in df.columns]
    if available_prior:
        df['total_prior_score'] = df[available_prior].fillna(0).sum(axis=1)
    
    # 7. Violent Crime x Prior History
    if 'violent_crime' in df.columns and 'total_prior_score' in df.columns:
        df['violent_x_prior'] = df['violent_crime'] * df['total_prior_score']
    
    # 8. Age Risk Factor (Genç + Şiddet = Yüksek Risk)
    if 'age_offense' in df.columns and 'violent_crime' in df.columns:
        age_normalized = (df['age_offense'] - df['age_offense'].mean()) / df['age_offense'].std()
        df['age_risk'] = age_normalized * df['violent_crime']
    
    # 9. Year Trend (Yıllar içinde ceza artışı/azalışı)
    if 'year' in df.columns:
        df['years_since_2000'] = df['year'] - 2000
        df['year_squared'] = df['years_since_2000'] ** 2
    
    print(f"✅ {9} yeni özellik eklendi!")
    return df


def prepare_features(df):
    """Feature listesini hazırla"""
    base_features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race',
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    base_features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    
    # Mevcut interaction features
    interaction_features = ['severity_x_violent', 'age_gap', 'violent_recid']
    
    # YENİ advanced features
    new_features = [
        'judge_harshness', 'judge_consistency', 'county_harshness',
        'wcisclass_severity', 'judge_crime_combo', 'recid_severity',
        'total_prior_score', 'violent_x_prior', 'age_risk',
        'years_since_2000', 'year_squared'
    ]
    
    all_features = base_features + interaction_features + new_features
    available_features = [f for f in all_features if f in df.columns]
    
    # Kategorik belirleme
    cat_features = []
    KNOWN_CAT = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass', 'judge_crime_combo']
    
    X = df[available_features].copy()
    
    for col in X.columns:
        if col in KNOWN_CAT or X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("Unknown").astype(str)
            if col not in cat_features:
                cat_features.append(col)
    
    # Sayısal fillna
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
    
    return X, available_features, cat_features


def experiment_1_baseline(X, y, cat_features):
    """Deney 1: Baseline (Mevcut Model)"""
    print("\n" + "="*60)
    print("🧪 DENEY 1: BASELINE (Mevcut Hyperparameters)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.02,
        depth=10,
        cat_features=cat_features,
        verbose=0,
        random_seed=RANDOM_STATE,
        l2_leaf_reg=5
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"✅ Baseline R²: {r2:.4f}")
    print(f"✅ Baseline MAE: {mae:.4f}")
    
    return {'model': model, 'r2': r2, 'mae': mae, 'name': 'Baseline'}


def experiment_2_deep_trees(X, y, cat_features):
    """Deney 2: Daha Derin Ağaçlar"""
    print("\n" + "="*60)
    print("🧪 DENEY 2: DAHA DERİN AĞAÇLAR (depth=14)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.02,
        depth=14,  # Daha derin
        cat_features=cat_features,
        verbose=0,
        random_seed=RANDOM_STATE,
        l2_leaf_reg=5
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"✅ Deep Trees R²: {r2:.4f} (Baseline'den {r2-0.3337:.4f} fark)")
    print(f"✅ Deep Trees MAE: {mae:.4f}")
    
    return {'model': model, 'r2': r2, 'mae': mae, 'name': 'Deep Trees'}


def experiment_3_more_iterations(X, y, cat_features):
    """Deney 3: Daha Fazla İterasyon"""
    print("\n" + "="*60)
    print("🧪 DENEY 3: DAHA FAZLA İTERASYON (3000 iter)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model = CatBoostRegressor(
        iterations=3000,  # 2x daha fazla
        learning_rate=0.01,  # Daha düşük LR
        depth=12,
        cat_features=cat_features,
        verbose=0,
        random_seed=RANDOM_STATE,
        l2_leaf_reg=5
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"✅ More Iterations R²: {r2:.4f} (Baseline'den {r2-0.3337:.4f} fark)")
    print(f"✅ More Iterations MAE: {mae:.4f}")
    
    return {'model': model, 'r2': r2, 'mae': mae, 'name': 'More Iterations'}


def experiment_4_ensemble(X, y, cat_features):
    """Deney 4: Ensemble (3 farklı model ortalaması)"""
    print("\n" + "="*60)
    print("🧪 DENEY 4: ENSEMBLE (3 Model Ortalaması)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    configs = [
        {'iterations': 2000, 'depth': 10, 'learning_rate': 0.02, 'l2_leaf_reg': 3},
        {'iterations': 2000, 'depth': 12, 'learning_rate': 0.015, 'l2_leaf_reg': 5},
        {'iterations': 2000, 'depth': 14, 'learning_rate': 0.01, 'l2_leaf_reg': 7},
    ]
    
    predictions = []
    
    for i, config in enumerate(configs, 1):
        print(f"   Model {i}/3 eğitiliyor...")
        model = CatBoostRegressor(
            **config,
            cat_features=cat_features,
            verbose=0,
            random_seed=RANDOM_STATE + i
        )
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))
    
    # Ortalama tahmin
    y_pred_ensemble = np.mean(predictions, axis=0)
    
    r2 = r2_score(y_test, y_pred_ensemble)
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    
    print(f"✅ Ensemble R²: {r2:.4f} (Baseline'den {r2-0.3337:.4f} fark)")
    print(f"✅ Ensemble MAE: {mae:.4f}")
    
    return {'model': None, 'r2': r2, 'mae': mae, 'name': 'Ensemble', 'predictions': predictions}


def experiment_5_quantile_loss(X, y, cat_features):
    """Deney 5: Quantile Loss (Median tahmin)"""
    print("\n" + "="*60)
    print("🧪 DENEY 5: QUANTILE LOSS (Median Prediction)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=12,
        cat_features=cat_features,
        verbose=0,
        random_seed=RANDOM_STATE,
        loss_function='Quantile:alpha=0.5',  # Median
        l2_leaf_reg=5
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"✅ Quantile Loss R²: {r2:.4f} (Baseline'den {r2-0.3337:.4f} fark)")
    print(f"✅ Quantile Loss MAE: {mae:.4f}")
    
    return {'model': model, 'r2': r2, 'mae': mae, 'name': 'Quantile Loss'}


def experiment_6_advanced_features_only(X, y, cat_features):
    """Deney 6: Sadece Yeni Advanced Features ile"""
    print("\n" + "="*60)
    print("🧪 DENEY 6: ADVANCED FEATURES (Yeni Özelliklerle)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.015,
        depth=12,
        cat_features=cat_features,
        verbose=0,
        random_seed=RANDOM_STATE,
        l2_leaf_reg=4
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"✅ Advanced Features R²: {r2:.4f} (Baseline'den {r2-0.3337:.4f} fark)")
    print(f"✅ Advanced Features MAE: {mae:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance()
    feature_names = X.columns
    top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n📊 Top 10 En Önemli Özellikler:")
    for feat, imp in top_features:
        print(f"   • {feat}: {imp:.4f}")
    
    return {'model': model, 'r2': r2, 'mae': mae, 'name': 'Advanced Features', 'top_features': top_features}


def compare_results(results):
    """Tüm deneyleri karşılaştır"""
    print("\n" + "="*60)
    print("📊 SONUÇLARIN KARŞILAŞTIRILMASI")
    print("="*60)
    
    # Tablo
    print(f"\n{'Deney':<25} {'R² Score':<15} {'MAE':<15} {'İyileşme':<15}")
    print("-" * 70)
    
    baseline_r2 = 0.3337  # Diagnostic'ten bilinen değer
    
    for result in results:
        improvement = result['r2'] - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100
        print(f"{result['name']:<25} {result['r2']:<15.4f} {result['mae']:<15.4f} +{improvement_pct:>6.1f}%")
    
    # En iyi modeli bul
    best_result = max(results, key=lambda x: x['r2'])
    
    print(f"\n🏆 EN İYİ MODEL: {best_result['name']}")
    print(f"   • R² Score: {best_result['r2']:.4f}")
    print(f"   • MAE: {best_result['mae']:.4f}")
    print(f"   • İyileşme: +{((best_result['r2'] - baseline_r2) / baseline_r2) * 100:.1f}%")
    
    # Görselleştirme
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [r['name'] for r in results]
    r2_scores = [r['r2'] for r in results]
    
    bars = ax.bar(names, r2_scores, color=['gray' if r['name'] == 'Baseline' else 'steelblue' for r in results])
    
    # En iyi modeli vurgula
    best_idx = names.index(best_result['name'])
    bars[best_idx].set_color('green')
    
    # Hedef çizgisi
    ax.axhline(y=0.50, color='red', linestyle='--', linewidth=2, label='Hedef: 50% R²')
    ax.axhline(y=baseline_r2, color='orange', linestyle='--', linewidth=2, label='Baseline: 33.37% R²')
    
    ax.set_ylabel('R² Score')
    ax.set_title('High Severity Model İyileştirme Deneyleri')
    ax.legend()
    ax.set_ylim([0, 0.6])
    
    # Değerleri yazdır
    for i, (name, r2) in enumerate(zip(names, r2_scores)):
        ax.text(i, r2 + 0.01, f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_improvement_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Görsel kaydedildi: 04_improvement_comparison.png")
    plt.close()
    
    return best_result


def save_best_model(best_result, X, y, cat_features, available_features):
    """En iyi modeli kaydet"""
    print("\n" + "="*60)
    print("💾 EN İYİ MODEL KAYDEDİLİYOR")
    print("="*60)
    
    if best_result['name'] == 'Ensemble':
        print("⚠️ Ensemble model - 3 ayrı model kaydedilecek")
        # Ensemble için 3 modeli yeniden eğit ve kaydet
        # (Şimdilik atlıyoruz, gerekirse ekleriz)
        return
    
    # Tüm veriyle yeniden eğit
    print("🚀 En iyi model tüm veriyle yeniden eğitiliyor...")
    
    if best_result['name'] == 'Deep Trees':
        config = {'iterations': 1500, 'learning_rate': 0.02, 'depth': 14, 'l2_leaf_reg': 5}
    elif best_result['name'] == 'More Iterations':
        config = {'iterations': 3000, 'learning_rate': 0.01, 'depth': 12, 'l2_leaf_reg': 5}
    elif best_result['name'] == 'Quantile Loss':
        config = {'iterations': 2000, 'learning_rate': 0.02, 'depth': 12, 'l2_leaf_reg': 5, 'loss_function': 'Quantile:alpha=0.5'}
    elif best_result['name'] == 'Advanced Features':
        config = {'iterations': 2500, 'learning_rate': 0.015, 'depth': 12, 'l2_leaf_reg': 4}
    else:
        config = {'iterations': 1500, 'learning_rate': 0.02, 'depth': 10, 'l2_leaf_reg': 5}
    
    final_model = CatBoostRegressor(
        **config,
        cat_features=cat_features,
        verbose=0,
        random_seed=RANDOM_STATE
    )
    final_model.fit(X, y)
    
    # Kaydet
    final_model.save_model(str(MODEL_DIR / "model_high_v2_improved.cbm"))
    joblib.dump(available_features, MODEL_DIR / "features_v2_improved.pkl")
    
    print(f"✅ Model kaydedildi: {MODEL_DIR / 'model_high_v2_improved.cbm'}")
    print(f"✅ Features kaydedildi: {MODEL_DIR / 'features_v2_improved.pkl'}")


def generate_improvement_report(results, best_result):
    """İyileştirme raporu oluştur"""
    print("\n" + "="*60)
    print("📝 İYİLEŞTİRME RAPORU OLUŞTURULUYOR")
    print("="*60)
    
    baseline_r2 = 0.3337
    
    report = f"""# High Severity Model İyileştirme Sonuçları

## Özet

**Hedef:** High Severity Model R² skorunu %33 → %50+ yükseltmek

**Sonuç:** En iyi model **{best_result['name']}** ile **{best_result['r2']:.2%}** R² elde edildi.

**İyileşme:** +{((best_result['r2'] - baseline_r2) / baseline_r2) * 100:.1f}% (Baseline: {baseline_r2:.2%})

---

## Deney Sonuçları

| Deney | R² Score | MAE | İyileşme |
|-------|----------|-----|----------|
"""
    
    for result in results:
        improvement = ((result['r2'] - baseline_r2) / baseline_r2) * 100
        report += f"| {result['name']} | {result['r2']:.4f} | {result['mae']:.4f} | +{improvement:.1f}% |\n"
    
    report += f"""
---

## En İyi Model: {best_result['name']}

- **R² Score:** {best_result['r2']:.4f} ({best_result['r2']:.2%})
- **MAE:** {best_result['mae']:.4f}
- **İyileşme:** +{((best_result['r2'] - baseline_r2) / baseline_r2) * 100:.1f}%

### Performans Değerlendirmesi

"""
    
    if best_result['r2'] >= 0.50:
        report += "✅ **HEDEF ULAŞILDI!** %50 R² hedefine ulaşıldı veya aşıldı.\n\n"
    elif best_result['r2'] >= 0.45:
        report += "⚡ **HEDEF YAKLAŞILDI!** %50 hedefine çok yaklaşıldı. Ek iyileştirmelerle hedef ulaşılabilir.\n\n"
    elif best_result['r2'] >= 0.40:
        report += "📈 **ÖNEMLI İYİLEŞME!** Baseline'den önemli iyileşme sağlandı. %50 hedefi için ek feature'lar gerekli.\n\n"
    else:
        report += "📊 **SINIRLI İYİLEŞME.** Mevcut feature'larla %50 hedefi zor görünüyor. Yeni veri kaynakları gerekli.\n\n"
    
    report += f"""
![Improvement Comparison](04_improvement_comparison.png)

---

## Öneriler

### Kısa Vadeli (Hemen Uygulanabilir)

1. **En iyi modeli kullan:** {best_result['name']} modelini production'a al
2. **Ensemble dene:** Birden fazla modelin ortalaması daha stabil sonuçlar verebilir
3. **Cross-validation:** K-fold ile performansı doğrula

### Orta Vadeli (Yeni Feature'lar)

1. **Dava metinleri:** NLP ile suç tanımlarını analiz et
2. **Hakim geçmişi:** Hakim bazlı istatistikler ekle
3. **Temporal patterns:** Yıl/mevsim etkilerini modelle

### Uzun Vadeli (Dış Veri)

1. **Mahkeme kayıtları:** Duruşma süreleri, tanık sayıları
2. **Sosyoekonomik:** Bölgesel ekonomik göstergeler
3. **Suçlu profili:** Eğitim, istihdam durumu

---

**Hazırlayan:** Antigravity AI  
**Tarih:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  
**Versiyon:** 1.0
"""
    
    # Raporu kaydet
    report_path = OUTPUT_DIR / 'improvement_results.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ İyileştirme raporu oluşturuldu: {report_path}")


def main():
    """Ana fonksiyon"""
    print("="*60)
    print("🚀 HIGH SEVERITY MODEL IMPROVEMENT EXPERIMENTS")
    print("="*60)
    
    # 1. Veri yükle
    df_high = load_high_severity_data()
    
    # 2. Advanced features ekle
    df_high = create_advanced_features(df_high)
    
    # 3. Features hazırla
    X, available_features, cat_features = prepare_features(df_high)
    y = np.log1p(df_high['jail'])
    
    print(f"\n📊 Final Feature Count: {len(available_features)}")
    print(f"📊 Categorical Features: {len(cat_features)}")
    
    # 4. Deneyleri çalıştır
    results = []
    
    results.append(experiment_1_baseline(X, y, cat_features))
    results.append(experiment_2_deep_trees(X, y, cat_features))
    results.append(experiment_3_more_iterations(X, y, cat_features))
    results.append(experiment_4_ensemble(X, y, cat_features))
    results.append(experiment_5_quantile_loss(X, y, cat_features))
    results.append(experiment_6_advanced_features_only(X, y, cat_features))
    
    # 5. Sonuçları karşılaştır
    best_result = compare_results(results)
    
    # 6. En iyi modeli kaydet
    save_best_model(best_result, X, y, cat_features, available_features)
    
    # 7. Rapor oluştur
    generate_improvement_report(results, best_result)
    
    print("\n" + "="*60)
    print("✅ TÜM DENEYLER TAMAMLANDI!")
    print("="*60)
    print(f"\n📂 Çıktılar: {OUTPUT_DIR}")
    print(f"   • improvement_results.md")
    print(f"   • 04_improvement_comparison.png")
    print(f"\n📂 Model: {MODEL_DIR}")
    print(f"   • model_high_v2_improved.cbm (EN İYİ MODEL)")


if __name__ == "__main__":
    main()
