"""
🚀 STEP 34 - ROUTER PRECISION OPTIMIZATION
============================================
Router modelinin 'Paranoyak' yapısını azaltarak (False Positive'leri düşürerek)
Ağır Suçlar sınıfındaki Precision (Kesinlik) değerini artırmayı hedefler.
Burada Threshold (Eşik Değeri) ayarlaması veya Sınıf Ağırlıkları (Class Weights) ile oynanır.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_curve
import joblib
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("../model_data_v2_interactions")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD = 3000

def create_router_features(df):
    df = df.copy()
    
    if 'highest_severity' in df.columns and 'violent_crime' in df.columns:
        df['severity_x_violent'] = df['highest_severity'] * df['violent_crime']
        
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        age_j = df['age_judge'].fillna(df['age_judge'].mean())
        age_o = df['age_offense'].fillna(df['age_offense'].mean())
        df['age_gap'] = age_j - age_o
        
    if 'is_recid_new' in df.columns and 'violent_crime' in df.columns:
        df['violent_recid'] = df['is_recid_new'] * df['violent_crime']
        
    if 'highest_severity' in df.columns and 'is_recid_new' in df.columns:
        df['severity_x_recid'] = df['highest_severity'] * df['is_recid_new']
        
    return df

def optimize_router_precision():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    df['target_class'] = (df['jail'] > THRESHOLD).astype(int)
    
    print("\n📊 Sınıf Dağılımı (0: <=3000, 1: >3000):")
    print(df['target_class'].value_counts(normalize=True))
    
    base_features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    base_features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    
    df = create_router_features(df)
    
    new_features = ['severity_x_violent', 'age_gap', 'violent_recid', 'severity_x_recid']
    base_features.extend(new_features)
    
    available_features = [f for f in base_features if f in df.columns]
    
    X = df[available_features].copy()
    y = df['target_class']
    
    cat_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("Unknown").astype(str)
            cat_features.append(col)
            
    for col in ['judge_id', 'county', 'zip']:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("Unknown")
            if col not in cat_features: cat_features.append(col)
            
    for col in X.columns:
        if col not in cat_features:
            X[col] = X[col].fillna(X[col].mean())
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n🚀 Precision Odaklı Router Modeli Eğitiliyor (Class Weights Ayarı)")
    
    # Eskiden auto_class_weights='Balanced' diyorduk. Bu, azınlık sınıfına çok ağırlık veriyordu.
    # Şimdi scale_pos_weight kullanıp biraz daha "katı/seçici" bir model tasarlıyoruz.
    # Sınıf dağılımı: Sınıf 0 (%92.5), Sınıf 1 (%7.5). Oran ~ 12.3
    # Mükemmel denge: 12.3 verirsek recall coşar precision düşer (Eski durum).
    # Biz precision'ı artırmak için bu ağırlığı 12.3'ten örn. 5'e çekiyoruz.
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.08,
        depth=6,
        cat_features=cat_features,
        scale_pos_weight=5.0, # Anahtar değişiklik! Aşırı dengeden biraz taviz veriyoruz.
        verbose=100,
        random_seed=42,
        eval_metric='F1',
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    print("\n📊 Precision-Odaklı Router Performansı:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report (Standart Eşik - 0.5):")
    print(classification_report(y_test, y_pred))
    
    print("\n⚖️ EŞİK DEĞERİ (THRESHOLD) OPTİMİZASYONU...")
    # Precision-Recall Curve ile en ideal kesim noktasını bulalım
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # F1 skorunu maksimize eden eşik değeri
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\n🥇 En İyi F1 Skoru için Optimal Eşik: {best_threshold:.4f}")
    
    # Özel bir kural koyalım: "Precision en az %60 olsun ama Recall da çok düşmesin"
    # Precision'ın %60'ı geçtiği ilk noktayı bulalım
    target_precision_idx = np.where(precisions >= 0.60)[0]
    
    if len(target_precision_idx) > 0:
        custom_idx = target_precision_idx[0]
        custom_threshold = thresholds[custom_idx] if custom_idx < len(thresholds) else thresholds[-1]
        
        print(f"🎯 Precision >= %60 Olması İçin Gereken Eşik: {custom_threshold:.4f}")
        
        # Yeni eşikle tahmin yap
        custom_preds = (y_prob >= custom_threshold).astype(int)
        print(f"\nClassification Report (Özel Eşik - {custom_threshold:.4f}):")
        print(classification_report(y_test, custom_preds))
        
        # Seçim senin: Karar verilen eşik değerini kaydediyoruz.
        print("\n💾 Precision Dengeli Router Modeli Kaydediliyor...")
        model.save_model(str(MODEL_DIR / "router_classifier_precision.cbm"))
        
        # Model class'larına eşik değerini de gömelim
        model.set_probability_threshold(custom_threshold)
        model.save_model(str(MODEL_DIR / "router_classifier_precision_with_threshold.cbm"))
        
    else:
        print("Model %60 Precision seviyesine ulaşamıyor.")

if __name__ == "__main__":
    optimize_router_precision()
