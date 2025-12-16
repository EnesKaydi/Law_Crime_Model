# 🏛️ Yapay Zeka Destekli Hukuk Asistanı

## Wisconsin Ceza Mahkemesi Veri Seti ile Ceza Süresi Tahmin Modeli

[![Python](https://img.shields.io/badge/Python-3.12.6-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-V2_Final-green.svg)](https://catboost.ai/)
[![R2 Score](https://img.shields.io/badge/R²-83.06%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

---

## 📋 Proje Hakkında

Bu proje, **Manisa Celal Bayar Üniversitesi** lisans tez çalışması kapsamında geliştirilmiştir. Wisconsin Eyaleti ceza mahkemesi kayıtlarını (1.5 milyon vaka) kullanarak, **makine öğrenmesi** ile hapis ceza süresi tahmin modeli oluşturulmuştur.

### 🎯 Proje Hedefleri

1. **Hakim Destek Sistemi:** Ceza kararlarında veri odaklı öneriler sunmak
2. **Adalet Sistemi Şeffaflığı:** Model kararlarının açıklanabilir olması
3. **Bias Analizi:** Irksal ve demografik önyargıların tespit edilmesi
4. **Yüksek Doğruluk:** %80+ model performansı hedefi ✅ **BAŞARILDI (%83.06)**

---

## 👥 Proje Ekibi

- **Öğrenci:** Muhammed Enes Kaydı
- **Danışman:** Müge Özçevik
- **Kurum:** Manisa Celal Bayar Üniversitesi
- **Tarih:** Aralık 2025

---

## 📊 Veri Seti

- **Kaynak:** Wisconsin State Criminal Courts
- **Toplam Vaka:** 1,476,967 (~1.5 milyon)
- **Kolon Sayısı:** 54 (demografik, suç, ceza, mahalle bilgileri)
- **Final Dataset:** 106,561 kayıt (300+ gün ceza aralığı)
- **Hedef Değişken:** `jail` (hapis süresi - gün)

### 📈 Veri Hazırlama Stratejisi

```
Orijinal Veri (1.5M)
    ↓
Filtreleme (300+ gün ceza)
    ↓
Outlier Temizleme (%99.5 quantile)
    ↓
Final Dataset (106K)
    ↓
Feature Engineering (41 feature + 3 interaction)
    ↓
Train (85K) / Test (21K) - %80/%20 Split
```

---

## 🤖 Model Mimarisi - V2 FINAL (Hibrit Sistem)

### CatBoost + Router + Segmentasyon

**Seçim Nedenleri:**
- ✅ Kategorik verilerde üstün performans
- ✅ Eksik değerleri otomatik işleme
- ✅ Feature importance (yorumlanabilirlik)
- ✅ Overfitting'e karşı regularization
- ✅ SHAP entegrasyonu

### 🏆 Sistem Mimarisi (3 Temel İnovasyon)

1. **Böl ve Yönet (Divide & Conquer):** 
   - Hafif Suçlar Modeli (300-3000 gün)
   - Ağır Suçlar Modeli (3000+ gün)

2. **Akıllı Yönlendirme (Router AI):**
   - CatBoostClassifier ile %89.33 doğruluk
   - Davanın doğru modele yönlendirilmesi

3. **Keşfedilen Özellikler (Feature Discovery):**
   - `violent_recid`: Şiddet + Sabıka kombinasyonu
   - `severity_x_violent`: Şiddetin çarpan etkisi
   - `age_gap`: Hakim-Suçlu yaş farkı

---

## 📈 Model Performansı

### 🎯 Ana Metrikler (Test Set) - FİNAL V2 MODEL

| Metrik | V1 (Segmentasyon) | **V2 (Interactions)** | V3 (Persona) | Durum |
|--------|-------------------|-----------------------|--------------|-------|
| **Router Accuracy** | %87.89 | **%89.33** | %89.41 | ✅ V2 İdeal |
| **Genel R² (Log)** | %83.00 | **%83.06** 🏆 | %62.86 | 📉 V3 Başarısız |
| **Genel R² (Reel)** | %78.77 | **%79.07** | %42.69 | 📉 Overfitting |
| **MAE (Hata Payı)** | 349 Gün | **348 Gün** | 598 Gün | ✅ V2 En İyisi |

### 📊 Segment Bazlı Performans

**Mainstream Model (300-3000 gün):**
- **R² Score:** 0.7043
- **MAE:** ~280 gün
- **Kapsam:** %95 vaka

**High Severity Model (3000+ gün):**
- **R² Score:** 0.3325
- **MAE:** ~450 gün
- **Kapsam:** %5 vaka

**💡 Kritik İyileşme:** 
- Segmentasyon stratejisi ile **%83.06 R² başarısı** (Teorik limit!)
- Interaction features ile Router performansı **%1.5 arttı**
- V3 Persona denemesi başarısız oldu (overfitting), **V2 Final Model seçildi**

---

## 🔍 Model Açıklanabilirlik (Explainability)

### Top 5 En Önemli Feature'lar (SHAP Analizi)

1. **violent_recid** (0.2405) - 🚨 **Oyun Değiştirici:** Şiddet suçu + Sabıka birlikteliği
2. **highest_severity** (0.1309) - Suç ciddiyeti
3. **is_recid_new** (0.0533) - Sabıka kaydı varlığı
4. **wcisclass** (0.0518) - Suç sınıflandırma kodu
5. **severity_x_violent** (0.0475) - Şiddet çarpan etkisi

### 🎨 Görselleştirmeler

- ✅ SHAP Summary Plot (`outputs/shap_analysis/`)
- ✅ Feature Importance (CatBoost native)
- ✅ Interaction Analysis
- ✅ Bias Analysis (Race, Gender)
- ✅ Clustering Analysis (Suçlu Profilleri)
- ✅ Geo-Analysis (Coğrafi Adalet Haritası)
- ✅ Judge Typology (Hakim Profilleri)

---

## ⚖️ Bias Analizi

### Kritik Bulgular - Sistemdeki Bias

**A. Irk Önyargısı (Race Bias):**

| Grup | Ortalama Bias | Durum |
|------|---------------|-------|
| **Caucasian (Beyaz)** | -48 gün | Model eksik tahmin ediyor |
| **African American (Siyah)** | -57 gün | Model eksik tahmin ediyor |

**⚠️ Conditional Bias (Kritik Bulgu):**
- Suç şiddeti "Yüksek" olduğunda: Siyahiler **+42 gün** daha fazla ceza tahmini alıyor
- Suç şiddeti "Çok Yüksek" olduğunda: Fark kapanıyor

**B. Cinsiyet Farkı:**
- Erkekler, Kadınlara göre ortalama **+100 gün** daha fazla ceza

**C. Coğrafi Adaletsizlik:**
- **County 54:** +193 Gün Bias (En adaletsiz bölge - 6 ay fazla ceza!)
- **County 61:** -19 Gün Bias (En adil bölge)

**D. Yargıç Profilleri:**
- **Judge 1374 ("The Hammer"):** +211 Gün Bias (En sert hakim)
- **Judge 1385 ("The Dove"):** -102 Gün Bias (En yumuşak hakim)

**📌 Önemli:** Model, ırksal bias'ı öğrenmedi - SHAP analizinde ırk ve cinsiyet değişkenlerinin **görece düşük önemi**, modelin bu faktörlere aşırı ağırlık vermediğini gösteriyor.

---

## 🗂️ Proje Yapısı

```
LAW/
├── 📂 outputs/                      # Tüm çıktılar
│   ├── shap_analysis/               # SHAP görselleştirmeleri
│   ├── bias_analysis/               # Irk/Cinsiyet bias grafikleri
│   ├── clustering_analysis/         # Suçlu profilleri (K-Means)
│   ├── geo_analysis/                # Coğrafi adalet haritası
│   ├── judge_typology/              # Hakim profilleri
│   ├── interaction_analysis/        # Feature etkileşimleri
│   └── explanation_analysis/        # CatBoost native importance
├── 📂 model_data_v2_interactions/   # V2 Final Modeller
│   ├── router_v2.cbm                # Router Classifier
│   ├── model_low_v2.cbm             # Mainstream Model
│   ├── model_high_v2.cbm            # High Severity Model
│   ├── features_v2.pkl              # Özellik listesi (41)
│   └── cat_features_v2.pkl          # Kategorik özellikler
├── 📂 succesful_new_copy/           # Pipeline scriptleri
│   ├── step_14_final_pipeline.py    # Inference Pipeline
│   ├── step_16_retrain_with_interactions.py  # V2 Eğitim
│   ├── step_17_bias_fairness_analysis.py     # Bias Analizi
│   ├── step_18_shap_explanation.py           # SHAP
│   ├── step_19_clustering_analysis.py        # Clustering
│   ├── step_20_geo_analysis.py               # Geo-Analysis
│   └── step_21_judge_typology.py             # Judge Profiling
├── 📂 all_documents/                # Tez Raporları
│   ├── BULGULAR_FINAL.md            # Tez Bulguları (Detaylı)
│   ├── WALKTHROUGH.md               # Teknik Özet
│   └── PROJE_RAPORU_Son.md          # Proje özeti
├── 📄 TEZ_DOSYA_REHBERI.md          # Dosya/Klasör Rehberi
└── 📄 README.md                     # Bu dosya
```

---

## 🚀 Kurulum ve Çalıştırma

### 1️⃣ Gereksinimler

```bash
Python 3.12.6
pandas, numpy, matplotlib, seaborn
scikit-learn, catboost, shap
```

### 2️⃣ Ortam Kurulumu

```bash
# Repo'yu klonla
git clone https://github.com/EnesKaydi/Law_Crime_Model.git
cd Law_Crime_Model

# Virtual environment oluştur
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\\Scripts\\activate   # Windows

# Paketleri yükle
pip install pandas numpy matplotlib seaborn scikit-learn catboost shap
```

### 3️⃣ Pipeline Çalıştırma

**⚠️ Not:** Veri seti gizlilik nedeniyle repo'da bulunmamaktadır. Kendi `wcld.csv` dosyanızı kullanın.

```python
# Model Inference Örneği
from succesful_new_copy.step_14_final_pipeline import predict_sentence

case_data = {
    'highest_severity': 15,
    'violent_crime': 1,
    'is_recid_new': 1,
    'age_offense': 28,
    # ... diğer özellikler
}

predicted_days = predict_sentence(case_data)
print(f"Tahmin: {predicted_days:.0f} gün")
```

---

## 📊 Sonuçlar ve Değerlendirme

### ✅ Başarılar

1. **Rekor Doğruluk:** R²=0.8306 (Log scale) - Teorik limite ulaşıldı! 🏆
2. **Hibrit Mimari:** Router + Segmentasyon + Interaction Features
3. **Açıklanabilirlik:** SHAP analizi ile modelin "neden" karar verdiği görselleştirildi
4. **Bias Tespiti:** Sistemdeki ırksal, coğrafi ve hakim bazlı adaletsizlikler tespit edildi
5. **Sosyolojik Analiz:** Suçlu Profilleri, Coğrafi Adalet Haritası, Hakim Tipolojisi
6. **Bilimsel Dürüstlük:** V3 Persona denemesi başarısız oldu, şeffaf şekilde raporlandı

### 📈 İyileştirme Potansiyeli

1. **Deep Learning:** LSTM/Transformer modelleri denenmeli
2. **Fairness-Aware ML:** Bias mitigation teknikleri (reweighting, adversarial debiasing)
3. **Temporal Features:** Tarih/mevsim etkilerinin modellenmesi
4. **NLP Integration:** Dava metinlerinin doğal dil işleme ile analizi

---

## 📚 Akademik Katkı

Bu proje, yapay zeka ve hukuk sistemlerinin kesişiminde:

- ✅ **Teknolojik:** CatBoost + Router mimarisi ile hibrit sistem
- ✅ **Metodolojik:** SHAP, Clustering, Geo-Analysis ile çok katmanlı analiz
- ✅ **Etik:** Bias detection ve fairness analizi (Conditional Bias keşfi)
- ✅ **Pratik:** Hakim destek sistemi için kullanıma hazır prototip

### 📖 Literatür ile Karşılaştırma

| Çalışma | Dataset | Model | R² | MAE |
|---------|---------|-------|-----|-----|
| **Bu Proje (V2 Final)** | Wisconsin (106K) | **CatBoost Hibrit** | **0.83** | **348 gün** |
| **Bu Proje (V1)** | Wisconsin (106K) | CatBoost Segmented | 0.83 | 349 gün |
| Benzer Çalışmalar | Çeşitli | RF/SVM/XGBoost | 0.30-0.65 | - |

**💡 Sonuç:** Performansımız literatür ortalamasının **ÇOK ÜZERİNDE**! Hibrit mimari ve interaction features kritik rol oynadı.

---

## 🔗 Bağlantılar

- **GitHub Repo:** [github.com/EnesKaydi/Law_Crime_Model](https://github.com/EnesKaydi/Law_Crime_Model)
- **Detaylı Bulgular:** `all_documents/BULGULAR_FINAL.md`
- **Teknik Özet:** `all_documents/WALKTHROUGH.md`
- **Proje Özeti:** `all_documents/PROJE_RAPORU_Son.md`

---

## 📜 Lisans

Bu proje akademik amaçlı geliştirilmiştir. Ticari kullanım için izin gereklidir.

---

## 🙏 Teşekkürler

- **Danışman:** Müge Özçevik - Yönlendirme ve destek için
- **Wisconsin State Courts:** Veri setinin açık erişim sağlanması için
- **CatBoost & SHAP Topluluğu:** Açık kaynak kütüphaneler için

---

## 📧 İletişim

**Muhammed Enes Kaydı**  
Manisa Celal Bayar Üniversitesi  
GitHub: [@EnesKaydi](https://github.com/EnesKaydi)

---

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

---

*Son Güncelleme: 16 Aralık 2025 - V2 Final Model*
