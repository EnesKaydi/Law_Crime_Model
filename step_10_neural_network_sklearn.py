
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/model_deep_learning_sklearn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("model_data_dl_sklearn")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_sklearn_nn():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Veri Hazırlığı
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    y = np.log1p(df['jail'])
    
    # Feature Engineering
    features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year',
        'wcisclass', 'release', 'max_hist_jail', 'pct_male', 'judge_id',
        'age_judge', 'age_offense', 'pct_black', 'sex', 'race', 
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail',
        'county', 'case_type', 'zip'
    ]
    prior_severity_cols = [c for c in df.columns if 'prior_charges_severity' in c]
    features.extend(prior_severity_cols)
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    
    # Kategorik ve Sayısal Ayrımı
    cat_features = []
    num_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("Unknown").astype(str)
            cat_features.append(col)
        else:
            X[col] = X[col].fillna(X[col].mean()) # Sayısal eksiklere ortalama
            num_features.append(col)
            
    # Judge ID, Zip string olmalı
    for col in ['judge_id', 'county', 'zip']:
        if col in X.columns and col not in cat_features:
            X[col] = X[col].astype(str).fillna("Unknown")
            if col not in cat_features: cat_features.append(col)
            if col in num_features: num_features.remove(col)

    print(f"📌 Kategorik: {len(cat_features)} | Sayısal: {len(num_features)}")

    # Encoding (Label Encoding - Neural Network için One-Hot daha iyi ama Cardinality çok yüksek)
    # Bu yüzden Label Encoding yapıp Scaler'a sokacağız (Embedding etkisi yaratmaya çalışıyoruz)
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    # Scale Etme (NN için Kritik!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Model (MLP Regressor) - Deep Learning
    print("\n brains Neural Network (MLP) Eğitiliyor...")
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), # 3 Derin Katman
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=200, # Epoch
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Değerlendirme
    print("\n📊 NN Sonuçları:")
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    r2_log = r2_score(y_test, y_pred_log)
    r2_orig = r2_score(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    
    print(f"🔹 R2 Score (Log Scale): {r2_log:.4f}")
    print(f"🔹 R2 Score (Original Scale): {r2_orig:.4f}")
    print(f"🔹 MAE: {mae:.2f} gün")
    
    # Kayıt
    joblib.dump(model, MODEL_DIR / "mlp_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "mlp_scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "mlp_encoders.pkl")
    
if __name__ == "__main__":
    train_sklearn_nn()
