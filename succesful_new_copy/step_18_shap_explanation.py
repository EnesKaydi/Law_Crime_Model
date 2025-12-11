
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import joblib
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("../model_data_v2_interactions")
OUTPUT_DIR = Path("../outputs/shap_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_shap():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı (Mainstream Model İçin: 300-3000 Gün)
    if 'jail' not in df.columns: return
    df = df[df['jail'].between(300, 3000)].copy()
    
    # Yeni Özellikleri Oluştur (V2 Modeli İçin Gerekli)
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        df['age_judge'] = df['age_judge'].fillna(df['age_judge'].mean())
        df['age_offense'] = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = df['age_judge'] - df['age_offense']
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
        
    print(f"✅ Analiz Verisi: {len(df)} satır (Mainstream Kitlesi)")
    
    # Modeli Yükle
    if not MODEL_DIR.exists():
        print("❌ Model klasörü yok!")
        return
        
    features = joblib.load(MODEL_DIR / "features_v2.pkl")
    cat_features = joblib.load(MODEL_DIR / "cat_features_v2.pkl")
    
    model = CatBoostRegressor()
    model.load_model(str(MODEL_DIR / "model_low_v2.cbm"))
    
    # SHAP için Örneklem (Tüm veri çok yavaş olur, 1000 örnek yeterli)
    # Özellikle Bias için Siyahi ve Beyaz dengeli bir örneklem alalım
    df_sample = df.groupby('race', group_keys=False).apply(lambda x: x.sample(min(len(x), 200))).sample(1000, random_state=42)
    
    X_sample = df_sample[features].copy()
    
    # Kategorik düzenleme
    for col in cat_features:
        if col in X_sample.columns:
            X_sample[col] = X_sample[col].fillna("Unknown").astype(str)
            
    # Sayısal eksik
    for col in X_sample.columns:
        if col not in cat_features:
            X_sample[col] = X_sample[col].fillna(X_sample[col].mean())
            
    print("⏳ SHAP Değerleri Hesaplanıyor (Biraz sürebilir)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # 1. SUMMARY PLOT (Genel Bakış)
    print("\n📊 Summary Plot çiziliyor...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("En Önemli Faktörler (SHAP)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png")
    
    # 2. SEVERITY INTERACTION (Şiddet Etkisi)
    # severity_x_violent özelliği nasıl çalışıyor?
    if 'severity_x_violent' in X_sample.columns:
        plt.figure()
        shap.dependence_plot("severity_x_violent", shap_values, X_sample, show=False)
        plt.title("Suç Şiddeti x Şiddet Eylemi Etkisi")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_severity_interaction.png")
        
    # 3. RACE BIAS DETAIL (Irk Etkisi)
    # Model 'race' değişkenine ne kadar önem veriyor?
    # Eğer 'race' importance listesinde en sonlardaysa, bias dolaylı demektir.
    # Eğer üstlerdeyse, model doğrudan ırkçı demektir.
    if 'race' in X_sample.columns:
        # One-hot yapmadan CatBoost ile çalıştığımız için 'race' tek sütun.
        # Categorical feature dependence plot
        plt.figure()
        shap.dependence_plot("race", shap_values, X_sample, show=False)
        plt.title("Irkın Modele Doğrudan Etkisi")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_race.png")
        
    print(f"\n💾 SHAP grafikleri kaydedildi: {OUTPUT_DIR}")
    
    # Önem Sıralamasını Yazdır
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_sample.columns, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    print("\n🏆 SHAP Feature Importance (Top 10):")
    print(feature_importance.head(10))

if __name__ == "__main__":
    analyze_shap()
