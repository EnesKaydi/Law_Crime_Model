
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_PATH = Path("model_data_advanced/catboost_model.cbm")
FEATURES_PATH = Path("model_data_advanced/features_list.pkl")
CAT_FEATURES_PATH = Path("model_data_advanced/cat_features_list.pkl")
OUTPUT_DIR = Path("outputs/segment_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_segments():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı (Aynı filtreler)
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    print(f"✅ Analiz Verisi: {df.shape[0]} satır")
    
    # Modeli Yükle
    if not MODEL_PATH.exists():
        print("⚠️ Model bulunamadı!")
        return
        
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    feature_names = joblib.load(FEATURES_PATH)
    cat_features = joblib.load(CAT_FEATURES_PATH)
    
    # Tahmin Hazırlığı
    X = df[feature_names].copy()
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
            X.loc[X[col] == 'nan', col] = "Unknown"
            
    # Tahmin
    print("⏳ Model tahminleri alınıyor...")
    y_pred_log = model.predict(X)
    df['prediction'] = np.expm1(y_pred_log)
    df['error'] = df['prediction'] - df['jail']
    df['abs_error'] = df['error'].abs()
    
    # 1. SEGMENTASYON (CEZA SÜRESİNE GÖRE)
    # Kullanıcının hipotezi: Belki 300-1000 arası iyidir ama yukarısı bozuktur?
    bins = [300, 1000, 2000, 3000, 5000, 10000, 99999]
    labels = ['300-1000', '1000-2000', '2000-3000', '3000-5000', '5000-10000', '10000+']
    
    df['segment'] = pd.cut(df['jail'], bins=bins, labels=labels)
    
    print("\n📊 SEGMENT BAZLI PERFORMANS ANALİZİ:")
    print("-" * 60)
    print(f"{'Segment':<12} | {'Adet':<6} | {'MAE (Gün)':<10} | {'Ort. Ceza':<10} | {'Hata Oranı(%)':<12}")
    print("-" * 60)
    
    segment_stats = []
    
    for label in labels:
        subset = df[df['segment'] == label]
        if len(subset) == 0: continue
        
        mae = mean_absolute_error(subset['jail'], subset['prediction'])
        mean_jail = subset['jail'].mean()
        error_pct = (mae / mean_jail) * 100
        count = len(subset)
        
        # R2 Score (Segment bazlı R2 bazen yanıltıcı olabilir ama bakalım)
        r2 = r2_score(subset['jail'], subset['prediction'])
        
        print(f"{label:<12} | {count:<6} | {mae:<10.1f} | {mean_jail:<10.1f} | %{error_pct:<10.1f}")
        segment_stats.append({'segment': label, 'mae': mae, 'count': count, 'error_pct': error_pct})

    # Grafikleştirelim
    seg_df = pd.DataFrame(segment_stats)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='segment', y='mae', data=seg_df, palette='viridis')
    plt.title('Segmentlere Göre Ortalama Hata (MAE)')
    plt.ylabel('Hata (Gün)')
    plt.savefig(OUTPUT_DIR / "segment_mae_analysis.png")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='segment', y='error_pct', data=seg_df, palette='magma')
    plt.title('Segmentlere Göre Oransal Hata (%)')
    plt.ylabel('Hata Oranı (%)')
    plt.savefig(OUTPUT_DIR / "segment_pct_analysis.png")
    
    # 2. SUÇ ŞİDDETİNE GÖRE (Severity Segment)
    print("\n⚖️ SUÇ ŞİDDETİ (Severity) BAZLI ANALİZ:")
    # Severity genelde 1-10 arası ama veride nasıl dağılmış?
    df['severity_bin'] = pd.cut(df['highest_severity'], bins=[0, 3, 6, 9, 20], labels=['Düşük (1-3)', 'Orta (4-6)', 'Yüksek (7-9)', 'Çok Yüksek (10+)'])
    
    print("-" * 60)
    print(f"{'Şiddet':<15} | {'Adet':<6} | {'MAE (Gün)':<10} | {'R2 Score':<10}")
    print("-" * 60)
    
    for label in ['Düşük (1-3)', 'Orta (4-6)', 'Yüksek (7-9)', 'Çok Yüksek (10+)']:
        subset = df[df['severity_bin'] == label]
        if len(subset) < 10: continue
        
        mae = mean_absolute_error(subset['jail'], subset['prediction'])
        r2 = r2_score(subset['jail'], subset['prediction'])
        print(f"{label:<15} | {len(subset):<6} | {mae:<10.1f} | {r2:<10.4f}")

    print(f"\n💾 Analiz grafikleri kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_segments()
