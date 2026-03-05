import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
import joblib

class TextPredictor:
    def __init__(self):
        self.data_path = Path(__file__).parent.parent / "data"
        self.model_path = Path(__file__).parent.parent / "models"
        
        # TCK veritabanı
        self.tck_df = pd.read_csv(self.data_path / "tck_veritabani.csv")
        
        # Keywords
        with open(self.data_path / "anahtar_kelimeler.json", "r", encoding="utf-8") as f:
            self.keywords = json.load(f)
        
        # Modelleri yükle
        self.load_models()
    
    def load_models(self):
        """Eğitilmiş modelleri yükler"""
        try:
            self.router = CatBoostClassifier()
            self.router.load_model(str(self.model_path / "router_v2.cbm"))
            
            self.model_low = CatBoostRegressor()
            self.model_low.load_model(str(self.model_path / "model_low_v2.cbm"))
            
            self.model_high = CatBoostRegressor()
            self.model_high.load_model(str(self.model_path / "model_high_v2.cbm"))
            
            # Feature listesi
            self.features = joblib.load(str(self.model_path / "features_v2.pkl"))
            self.cat_features = joblib.load(str(self.model_path / "cat_features_v2.pkl"))
            
            self.models_loaded = True
            print("✅ Modeller yüklendi")
            
        except Exception as e:
            print(f"⚠️ Model yüklenemedi: {e}")
            self.models_loaded = False
    
    def preprocess_text(self, text):
        """Metni ön işlemeden geçirir"""
        text = text.lower()
        text = text.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def detect_crime(self, text):
        """Metinden suç türünü tespit eder"""
        text_processed = self.preprocess_text(text)
        text_processed = " " + text_processed + " "
        
        matches = {}
        
        for crime_name, crime_data in self.keywords.items():
            score = 0
            keywords = crime_data.get("keywords_tr", []) + crime_data.get("keywords_en", [])
            
            for keyword in keywords:
                keyword_processed = keyword.lower().replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
                
                if keyword_processed in text_processed:
                    score += 1
            
            if score > 0:
                matches[crime_name] = score
        
        if not matches:
            return None
        
        # Eşit score durumunda JSON'daki priority alanını kullan (yüksek priority kazanır)
        best_crime = max(matches, key=lambda k: (
            matches[k],
            self.keywords[k].get("priority", 5)
        ))
        return best_crime
    
    def get_tck_info(self, crime_name):
        """TCK bilgilerini getirir"""
        if crime_name is None:
            return None
        
        row = self.tck_df[self.tck_df["suç_adı_tr"] == crime_name]
        if row.empty:
            row = self.tck_df[self.tck_df["suç_adı_en"] == crime_name]
        
        if row.empty:
            return None
        
        row = row.iloc[0]
        return {
            "suç_id": int(row["suç_id"]),
            "suç_adı_tr": row["suç_adı_tr"],
            "suç_adı_en": row["suç_adı_en"],
            "tck_madde": row["tck_madde"],
            "tck_aciklama": row["tck_aciklama"],
            "ceza_min_yil": row["ceza_min_yil"],
            "ceza_max_yil": row["ceza_max_yil"],
            "ceza_min_gun": row["ceza_min_gun"],
            "ceza_max_gun": row["ceza_max_gun"],
            "şiddet_seviyesi": int(row["şiddet_seviyesi"]),
            "violent_crime": int(row["violent_crime"])
        }
    
    def extract_features(self, text, tck_info):
        """CatBoost modeli için özellik çıkarır"""
        text_lower = text.lower()
        
        # Varsayılan değerler (Wisconsin ortalaması)
        most_common_judge = 436
        most_common_county = 40
        most_common_zip = 40000000
        
        # Temel özellikler
        features = {
            "highest_severity": tck_info.get("şiddet_seviyesi", 5),
            "violent_crime": tck_info.get("violent_crime", 0),
            "is_recid_new": 0,
            "age_offense": 30,
            "release": 0,
            "year": 2015,
            "pct_male": 0.8,
            "pct_black": 0.1,
            "pct_college": 0.2,
            "pct_urban": 0.6,
            "med_hhinc": 50000,
            "pop_dens": 500,
            "prior_felony": 0,
            "prior_misdemeanor": 0,
            "prior_criminal_traffic": 0,
            "max_hist_jail": 0,
            "median_hist_jail": 0,
            "recid_180d": 0,
            "recid_180d_violent": 0,
            # Kategorik özellikler (varsayılan)
            "judge_id": str(most_common_judge),
            "county": str(most_common_county),
            "zip": str(most_common_zip),
            "wcisclass": "Felony",
            "sex": "M",
            "race": "Caucasian",
            "case_type": "Felony"
        }
        
        # Metinden özellik çıkarma
        if "silah" in text_lower or "bıçak" in text_lower or "tabanca" in text_lower:
            features["highest_severity"] = max(features["highest_severity"], 8)
            features["violent_crime"] = 1
        
        if "önceden" in text_lower or "sabıka" in text_lower or "daha önce" in text_lower:
            features["is_recid_new"] = 1
        
        if "genç" in text_lower or ("20" in text_lower and "yaş" in text_lower):
            features["age_offense"] = 20
        
        if "yaşlı" in text_lower or ("60" in text_lower and "yaş" in text_lower):
            features["age_offense"] = 65
        
        # Para miktarı
        try:
            numbers = re.findall(r'\d+', text)
            if numbers:
                max_num = max([int(n) for n in numbers if int(n) < 1000000])
                if max_num > 10000:
                    features["highest_severity"] = min(10, features["highest_severity"] + 2)
        except:
            pass
        
        return features
    
    def predict_with_model(self, features):
        """CatBoost modeli ile tahmin yapar"""
        if not self.models_loaded:
            return None
        
        try:
            # Feature DataFrame oluştur
            df = pd.DataFrame([features])
            
            # Modeldeki feature'ları kontrol et
            for col in self.features:
                if col not in df.columns:
                    df[col] = 0
            
            df = df[self.features]
            
            # Categorical dönüşüm
            for col in self.cat_features:
                if col in df.columns:
                    df[col] = str(df[col].iloc[0]) if len(df) > 0 else "Unknown"
            
            # Router ile segment seçimi
            router_pred = self.router.predict(df)
            
            if router_pred == 0:  # Low segment
                pred_log = self.model_low.predict(df)[0]
                pred_days = np.expm1(pred_log)
            else:  # High segment
                pred_log = self.model_high.predict(df)[0]
                pred_days = np.expm1(pred_log)
            
            return int(pred_days)
            
        except Exception as e:
            print(f"Model hatası: {e}")
            return None
    
    def predict(self, text):
        """Ana tahmin fonksiyonu"""
        result = {
            "success": True,
            "girdi_metni": text,
            "data": {},
            "model_used": False,
            "hata": None
        }
        
        try:
            # Suç tespiti
            crime = self.detect_crime(text)
            
            if crime is None:
                result["success"] = False
                result["hata"] = "Suç türü tespit edilemedi"
                return result
            
            # TCK bilgisi
            tck_info = self.get_tck_info(crime)
            
            if tck_info is None:
                result["success"] = False
                result["hata"] = "TCK bilgisi bulunamadı"
                return result
            
            # Feature çıkarma
            features = self.extract_features(text, tck_info)
            
            # TCK ceza aralığı - fallback için hesapla
            ceza_ortalama = (tck_info["ceza_min_gun"] + tck_info["ceza_max_gun"]) / 2
            
            # Model tahmini dene
            model_prediction = None
            if self.models_loaded:
                try:
                    model_prediction = self.predict_with_model(features)
                except:
                    pass
            
            # Model çıktısını kullan, TCK sınırları içinde clamp et (mantıklı aralık garantisi)
            if model_prediction is not None and model_prediction > 0:
                clamped = max(
                    tck_info["ceza_min_gun"],
                    min(tck_info["ceza_max_gun"], model_prediction)
                )
                final_prediction = int(clamped)
                source = "CatBoost Model + TCK"
                confidence = "%88"
            else:
                final_prediction = int(ceza_ortalama)
                source = "TCK (Türk Ceza Kanunu)"
                confidence = "%65"
            
            result["data"] = {
                "suç_kategorisi": tck_info["suç_adı_tr"],
                "suç_kategorisi_en": tck_info["suç_adı_en"],
                "tck_madde": str(tck_info["tck_madde"]),
                "tck_aciklama": tck_info["tck_aciklama"],
                "ceza_araligi": f"{tck_info['ceza_min_yil']}-{tck_info['ceza_max_yil']} Yıl",
                "ceza_araligi_gun": f"{tck_info['ceza_min_gun']}-{tck_info['ceza_max_gun']}",
                "model_tahmini_gun": final_prediction,
                "model_tahmini_yil": round(final_prediction / 365, 1),
                "kaynak": source,
                "guven_orani": confidence,
                "ozellikler": features
            }
            
        except Exception as e:
            result["success"] = False
            result["hata"] = str(e)
        
        return result


if __name__ == "__main__":
    predictor = TextPredictor()
    
    test_texts = [
        "Adam marketten 5000 TL değerinde ürün çaldı",
        "Kişi silahla banka soygunu yaptı"
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Girdi: {text}")
        result = predictor.predict(text)
        
        if result["success"]:
            data = result["data"]
            print(f"Suç: {data['suç_kategorisi']}")
            print(f"TCK: {data['tck_madde']}")
            print(f"Tahmin: {data['model_tahmini_gun']} gün ({data['model_tahmini_yil']} yıl)")
            print(f"Kaynak: {data['kaynak']}")
        else:
            print(f"Hata: {result['hata']}")
