
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("../outputs/router_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("../model_data_router")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_router_classifier():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı (Aynı filtreler, ama Target Binary olacak)
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # THRESHOLD = 3000 Gün
    # 0: Hafif/Orta (Mainstream)
    # 1: Ağır (High Severity)
    THRESHOLD = 3000
    df['target_class'] = (df['jail'] > THRESHOLD).astype(int)
    
    # Sınıf Dağılımı
    print("\n📊 Sınıf Dağılımı (0: <=3000, 1: >3000):")
    print(df['target_class'].value_counts(normalize=True))
    print(f"Toplam Veri: {len(df)}")
    
    # Özellikler (Regresyonda kullanılanların aynısı)
    features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].copy()
    y = df['target_class']
    
    # Kategorik İşlemler
    cat_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("Unknown").astype(str)
            cat_features.append(col)
            
    for col in ['judge_id', 'county', 'zip']:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("Unknown")
            if col not in cat_features: cat_features.append(col)
            
    print(f"📌 Kategorik Değişkenler: {cat_features}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model Eğitimi (CatBoost Classifier)
    # Dengesiz veri olduğu için auto_class_weights='Balanced' kullanıyoruz
    print("\n🚀 Router Modeli Eğitiliyor...")
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        verbose=100,
        random_seed=42,
        eval_metric='F1', # Dengesiz veride Accuracy yerine F1 daha iyi
        auto_class_weights='Balanced',
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Değerlendirme
    print("\n📊 Router Performansı:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    print(f"🔹 Accuracy: %{acc*100:.2f}")
    print(f"🔹 F1 Score: %{f1*100:.2f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Görselleştirme
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Mainstream', 'High'], yticklabels=['Mainstream', 'High'])
    plt.title('Router Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    
    # Kayıt
    model.save_model(str(MODEL_DIR / "router_classifier.cbm"))
    joblib.dump(available_features, MODEL_DIR / "router_features.pkl")
    
    print(f"\n💾 Router Model Kaydedildi: {MODEL_DIR}")
    
    if f1 > 0.70:
        print("✅ Router başarıyla çalışıyor! Pipeline kurulabilir.")
    else:
        print("⚠️ Router performansı düşük, manuel yönlendirme gerekebilir.")

if __name__ == "__main__":
    train_router_classifier()
