# Web Entegrasyon Rehberi: Hukuk Asistanı

Bu rehber, eğitilen yapay zeka modelini (CatBoost) bir web uygulamasına (Flask/Django/FastAPI) nasıl entegre edeceğinizi anlatır.

## 📂 Gerekli Dosyalar
Web uygulamanızın çalışması için aşağıdaki dosyaları `model_data_advanced` klasöründen sunucuya taşımanız gerekir:

1.  `catboost_model.cbm` (Eğitilmiş Model)
2.  `features_list.pkl` (Modelin beklediği kolon isimleri)
3.  `cat_features_list.pkl` (Kategorik kolon listesi)

## 🚀 Örnek Kullanım (Backend)
`step_07_web_inference_example.py` dosyasında çalışan bir örnek mevcuttur. Aşağıda basit bir Flask API taslağı verilmiştir.

### Flask API Şablonu

```python
from flask import Flask, request, jsonify
from catboost import CatBoostRegressor
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Modeli Yükle (Uygulama Başlarken 1 Kez)
model = CatBoostRegressor()
model.load_model("model_data_advanced/catboost_model.cbm")
feature_names = joblib.load("model_data_advanced/features_list.pkl")
cat_features = joblib.load("model_data_advanced/cat_features_list.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json # JSON verisi al
        
        # DataFrame Hazırla
        row = {}
        for feat in feature_names:
            row[feat] = data.get(feat, np.nan) # Eksikse NaN
            
        df = pd.DataFrame([row])
        
        # Kategorik Dönüşüm (Zorunlu)
        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
                df.loc[df[col] == 'nan', col] = "Unknown"
                
        # Tahmin
        pred_log = model.predict(df)[0]
        days = np.expm1(pred_log)
        
        return jsonify({
            'ceza_gun': round(days),
            'ceza_yil': round(days / 365, 1),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
```

## 📩 Örnek İstek (JSON)
Hakim arayüzünden frontend'in göndermesi gereken JSON formatı:

```json
{
  "highest_severity": 10,
  "violent_crime": 1,
  "is_recid_new": 1,
  "judge_id": "673", 
  "sex": "Male",
  "age_offense": 34,
  "year": 2024
}
```

> [!TIP]
> **Hakim ID Seçimi:** Arayüzde hakimlerin ID'sini drop-down olarak koyabilirsiniz. Eğer yeni bir hakimse veya ID'si yoksa bu alanı göndermeyin, sistem otomatik olarak "Unknown" (Global Ortalama) kabul edecektir.

## ⚠️ Önemli Notlar
1.  **judge_id:** Model, `judge_id`'yi kategorik bir değişken olarak öğrendi. Eğer veritabanınızdaki hakim ID'leri ile modeldeki ID'ler eşleşiyorsa (örneğin "673" numaralı hakim veride varsa), model o hakimin geçmiş kararlarını bilir.
2.  **Eksik Veri:** Kullanıcı her alanı doldurmak zorunda değil. Doldurulmayan alanlar model tarafından "bilinmiyor" olarak işlenir ve en mantıklı ortalama tahmin üretilir.
