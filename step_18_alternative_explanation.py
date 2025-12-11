
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
import joblib
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("model_data_v2_interactions")
OUTPUT_DIR = Path("outputs/explanation_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_model_explanation():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı
    if 'jail' not in df.columns: return
    df = df[df['jail'].between(300, 3000)].copy()
    
    # Özellik Üretimi
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        df['age_judge'] = df['age_judge'].fillna(df['age_judge'].mean())
        df['age_offense'] = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = df['age_judge'] - df['age_offense']
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
        
    print(f"✅ Analiz Verisi: {len(df)} satır")
    
    # Model Yükle
    if not MODEL_DIR.exists():
        print("❌ Model klasörü yok!")
        return
        
    features = joblib.load(MODEL_DIR / "features_v2.pkl")
    cat_features = joblib.load(MODEL_DIR / "cat_features_v2.pkl")
    
    model = CatBoostRegressor()
    model.load_model(str(MODEL_DIR / "model_low_v2.cbm"))
    
    # 1. FEATURE IMPORTANCE (PredictionValuesChange)
    print("\n📊 Feature Importance Hesaplanıyor...")
    importance = model.get_feature_importance(type="PredictionValuesChange")
    feature_imp = pd.DataFrame({'feature': features, 'importance': importance})
    feature_imp = feature_imp.sort_values(by='importance', ascending=False)
    
    print("\n🏆 En Önemli 15 Faktör:")
    print(feature_imp.head(15))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_imp.head(20))
    plt.title('Model Kararlarını Etkileyen En Önemli Faktörler')
    plt.xlabel('Etki Gücü (%)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    
    # 2. INTERACTION ANALYSIS (CatBoost Internal)
    print("\n🔗 Özellik Etkileşimleri Hesaplanıyor...")
    
    # Pool nesnesi gerekli
    X = df[features].copy()
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
            
    pool = Pool(X, cat_features=cat_features)
    
    interaction = model.get_feature_importance(pool, type="Interaction")
    interaction_df = pd.DataFrame(interaction, columns=['feature_index_1', 'feature_index_2', 'strength'])
    
    # İsimleri eşleştir (Indexler float gelebilir, int'e çevir)
    interaction_df['feature_1'] = interaction_df['feature_index_1'].astype(int).apply(lambda x: features[x])
    interaction_df['feature_2'] = interaction_df['feature_index_2'].astype(int).apply(lambda x: features[x])
    interaction_df = interaction_df.sort_values(by='strength', ascending=False)
    
    print("\n🤝 En Güçlü Etkileşimler (Top 10):")
    print(interaction_df[['feature_1', 'feature_2', 'strength']].head(10))
    
    # 3. YENİ ÖZELLİKLERİN ETKİSİ
    print("\n🆕 Yeni Özelliklerin Performansı:")
    new_feats = ['severity_x_violent', 'age_gap', 'violent_recid']
    for f in new_feats:
        if f in feature_imp['feature'].values:
            rank = feature_imp[feature_imp['feature'] == f].index[0]
            score = feature_imp[feature_imp['feature'] == f]['importance'].values[0]
            print(f"   • {f}: Sıra #{rank+1}, Önem: {score:.2f}")
    
    print(f"\n💾 Grafikler kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_model_explanation()
