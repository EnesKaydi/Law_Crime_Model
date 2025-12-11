
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
tf.random.set_seed(42)

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("outputs/model_deep_learning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("model_data_dl")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_neural_network(numeric_features, cat_feature_info):
    # INPUTS
    inputs = []
    embeddings = []
    
    # KATEGORİK GİRDİLER (EMBEDDINGS)
    for name, vocab_size, embedding_dim in cat_feature_info:
        input_layer = layers.Input(shape=(1,), name=f"input_{name}")
        inputs.append(input_layer)
        
        embedding_layer = layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            name=f"embedding_{name}"
        )(input_layer)
        
        flat_embedding = layers.Flatten()(embedding_layer)
        embeddings.append(flat_embedding)
        
    # SAYISAL GİRDİLER
    num_input = layers.Input(shape=(len(numeric_features),), name="input_numeric")
    inputs.append(num_input)
    
    # BİRLEŞTİRME
    if embeddings:
        merged = layers.Concatenate()([num_input] + embeddings)
    else:
        merged = num_input
        
    # DENSE LAYERS (MLP)
    x = layers.Dense(256, activation='gelu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='gelu')(x)
    
    # ÇIKIŞ (REGRESYON)
    output = layers.Dense(1, name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber', # Huber loss outlier'lara karşı MSE'den daha dirençli
        metrics=['mae']
    )
    return model

def train_deep_learning():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # 1. VERİ HAZIRLIĞI
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    y = np.log1p(df['jail'])
    
    # 2. FEATURE ENGINEERING
    # Sayısal Özellikler
    numeric_features = [
        'highest_severity', 'violent_crime', 'is_recid_new', 'year', 'age_offense',
        'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic',
        'avg_hist_jail', 'median_hist_jail', 'min_hist_jail'
    ]
    numeric_features.extend([c for c in df.columns if 'prior_charges_severity' in c])
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    # Kategorik Özellikler (Embedding Yapılacaklar)
    # judge_id, county, zip, case_type, race, sex
    cat_features_targets = ['judge_id', 'county', 'zip', 'case_type', 'race', 'sex', 'wcisclass']
    cat_features = [c for c in cat_features_targets if c in df.columns]

    print(f"📊 Sayısal Özellikler ({len(numeric_features)}): {numeric_features}")
    print(f"📌 Kategorik Özellikler ({len(cat_features)}): {cat_features}")
    
    # Eksik Veri Doldurma (Sayısal için Mean, Kategorik için Unknown)
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].mean())
        
    for col in cat_features:
        df[col] = df[col].astype(str).replace('nan', 'Unknown').fillna('Unknown')
        
    X_num = df[numeric_features].values
    
    # Normalizasyon (Sayısal)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    
    # Label Encoding (Kategorik)
    encoders = {}
    X_cat_dict = {}
    cat_info = [] # (name, vocab_size, embedding_dim)
    
    for col in cat_features:
        le = LabelEncoder()
        # Bilinmeyenleri de öğrenmesi için fit
        encoded = le.fit_transform(df[col])
        encoders[col] = le
        X_cat_dict[f"input_{col}"] = encoded
        
        vocab_size = len(le.classes_)
        # Embedding Boyutu Kuralı: min(50, vocab_size/2)
        embedding_dim = min(50, (vocab_size + 1) // 2)
        cat_info.append((col, vocab_size, embedding_dim))
        
    # Bölme
    split_idx = int(len(df) * 0.8)
    
    X_train_num = X_num[:split_idx]
    X_test_num = X_num[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    train_inputs = {"input_numeric": X_train_num}
    test_inputs = {"input_numeric": X_test_num}
    
    for col in cat_features:
        train_inputs[f"input_{col}"] = X_cat_dict[f"input_{col}"][:split_idx]
        test_inputs[f"input_{col}"] = X_cat_dict[f"input_{col}"][split_idx:]
        
    # 3. MODEL KURMA
    print("\n🧠 Neural Network Kuruluyor...")
    model = build_neural_network(numeric_features, cat_info)
    # model.summary()
    
    # Callbackler
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    
    # 4. EĞİTİM
    print("🚀 Model Eğitiliyor...")
    history = model.fit(
        train_inputs, y_train,
        validation_data=(test_inputs, y_test),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 5. DEĞERLENDİRME
    print("\n📊 Deep Learning Sonuçları:")
    y_pred_log = model.predict(test_inputs).flatten()
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    r2_log = r2_score(y_test, y_pred_log)
    r2_orig = r2_score(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    
    print(f"🔹 R2 Score (Log Scale): {r2_log:.4f}")
    print(f"🔹 R2 Score (Original Scale): {r2_orig:.4f}")
    print(f"🔹 MAE: {mae:.2f} gün")
    print(f"🔹 RMSE: {rmse:.2f} gün")
    
    # Kayıt
    model.save(OUTPUT_DIR / "dl_model.keras")
    joblib.dump(scaler, MODEL_DIR / "dl_scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "dl_encoders.pkl")
    
    # Loss Grafiği
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss (Huber)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "training_loss.png")
    
    print(f"\n💾 Model ve grafikler kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    train_deep_learning()
