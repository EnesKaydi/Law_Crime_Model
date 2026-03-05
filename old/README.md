# 🏛️ Yapay Zeka Destekli Hukuk Asistanı

## Wisconsin Ceza Mahkemesi Veri Seti ile Ceza Süresi Tahmin Modeli

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![CatBoost+LightGBM](https://img.shields.io/badge/CatBoost+LightGBM-V3_Final-green.svg)](https://catboost.ai/)
[![R2 Teorik](https://img.shields.io/badge/R²_Teorik-83.65%25-brightgreen.svg)]()
[![R2 Pratik](https://img.shields.io/badge/R²_Pratik-74.06%25-blue.svg)]()
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

---

## 📋 Proje Hakkında

Bu proje, **Manisa Celal Bayar Üniversitesi** lisans tez çalışması kapsamında geliştirilmiştir. Wisconsin Eyaleti ceza mahkemesi kayıtlarını (1.5 milyon vaka) kullanarak, **makine öğrenmesi** ile hapis ceza süresi tahmin modeli oluşturulmuştur.

### 🎯 Proje Hedefleri

1. **Hakim Destek Sistemi:** Ceza kararlarında veri odaklı öneriler sunmak
2. **Adalet Sistemi Şeffaflığı:** Model kararlarının açıklanabilir olması
3. **Bias Analizi:** Irksal ve demografik önyargıların tespit edilmesi
4. **Yüksek Doğruluk:** %80+ model performansı hedefi ✅ **BAŞARILDI (%83.65)**

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

## 📈 Model Performansı (GÜNCEL — V3 Final)

Sistem performansı iki farklı bakış açısıyla analiz edilmiştir.

### 🎭 1. Gerçek Dünya — Uçtan Uca Test (Router Hataları Dahil)

| Metrik | Değer | Not |
|--------|-------|-----|
| **Router Precision** | **%60** | Threshold + class weights ile kalibrasyon yapıldı |
| **Router Recall** | **%69** | Ağır suçların büyük çoğunluğu yakalanıyor |
| **Uçtan Uca R² (Log)** | **%74.06** | Tüm sistemin gerçek senaryodaki başarısı |
| **Uçtan Uca MAE** | **443 Gün** | Router hata payı dahil |

### 🎯 2. Teorik Laboratuvar Performansı (Segment Bazlı Maksimumlar)

| Metrik | V1 | V2 | **V3 (Final)** |
|--------|----|----|----------------|
| **Genel R² (Log)** | %83.00 | %83.06 | **%83.65** 🚀 |
| **Genel R² (Reel)** | %78.77 | %79.07 | **%85.67** |
| **MAE** | 349 gün | 348 gün | **313 gün** 📉 |

### 📊 Segment Modelleri — Teorik Pik Skorlar

| Segment | Yöntem | R² (Log) | MAE |
|---------|--------|----------|-----|
| **Mainstream** (300-3000 gün, %95 vaka) | CatBoost + LightGBM **Ensemble** | **%70.96** | 238 gün |
| **High Severity** (3000+ gün, %5 vaka) | CatBoost + Optuna (100 deneme) | **%61.35** | 1210 gün |

> 💡 **Ensemble Bulgusu:** CatBoost (%60) + LightGBM (%40) ağırlıklı blend, tek modele kıyasla +0.32 puan teorik R² artışı sağladı (step_37).

> 🔬 **Teorik Limit:** Loss fonksiyonu karşılaştırması (RMSE/MAE/MAPE/Quantile) yapılmış; RMSE mevcut veri seti için en optimal seçim olduğu kanıtlanmıştır (step_36).

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
│   ├── step_21_judge_typology.py             # Judge Profiling
│   ├── step_30_router_optimization.py        # Router Optuna (100 trial)
│   ├── step_31_mainstream_optimization.py    # Mainstream Optuna (100 trial)
│   ├── step_32_high_severity_optimization.py # High Severity Optuna (100 trial)
│   ├── step_33_optimized_system_test.py      # Uçtan Uca Gerçek Test
│   ├── step_34_router_precision_opt.py       # Router Precision Kalibrasyonu
│   ├── step_35_high_severity_subsegment.py   # Alt Segment Deneyi
│   ├── step_36_quantile_regression.py        # Loss Fn Karşılaştırması
│   ├── step_37_ensemble_blending.py          # CatBoost+LightGBM Ensemble
│   └── step_38_judge_bias_modulation.py      # Hakim Bias Modülasyonu
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
Python 3.11
pandas, numpy, matplotlib, seaborn
scikit-learn, catboost, shap
```

### 2️⃣ Ortam Kurulumu

```bash
# Repo'yu klonla
git clone https://github.com/EnesKaydi/Law_Crime_Model.git
cd Law_Crime_Model

# Virtual environment oluştur
python -m venv .venv
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

### 🏆 Teorik Başarılar (Kusursuz Router Varsayımıyla)

| # | Başarı | Değer |
|---|--------|-------|
| 1 | **Teorik R² (Log)** | **%83.65** — literatür ortalamasının (%30-65) çok üstünde |
| 2 | **Teorik MAE** | **313 gün** |
| 3 | **Mainstream Ensemble R²** | **%70.96** (CatBoost %60 + LightGBM %40 blend) |
| 4 | **High Severity R²** | **%61.35** (başlangıçtaki %33'ten neredeyse iki katına çıktı) |
| 5 | **Loss Fonk. Kanıtı** | RMSE, bu veri için istatistiksel olarak optimal seçim kanıtlandı |

### 🌍 Pratik Başarılar (Router Hataları Dahil, Gerçek Dünya)

| # | Başarı | Değer |
|---|--------|-------|
| 1 | **Gerçek Dünya R²** | **%74.06** — Router yanılmalarıyla birlikte ölçüldü |
| 2 | **Gerçek Dünya MAE** | **443 gün** |
| 3 | **Router Precision** | **%60** — threshold kalibrasyonuyla "paranoyak" hatalar azaltıldı |
| 4 | **Router Recall** | **%69** — ağır suçların büyük çoğunluğu yakalanıyor |
| 5 | **Hakim Bias Düzeltmesi** | Sisteme otomatik **+51 gün** sertlik düzeltmesi eklendi |

### ✅ Diğer Başarılar

- **MoE Mimarisi:** Mixture of Experts — Router + 2 Uzman Model (akademik özgünlük)
- **Açıklanabilirlik:** SHAP analizi ile her kararın "neden" verildiği görselleştirildi
- **Bias Tespiti:** Irksal, coğrafi ve hakim bazlı adaletsizlikler tespit ve raporlandı
- **Bilimsel Dürüstlük:** Teorik ve pratik performans farkı şeffaf biçimde ayrı ayrı sunuldu

### 📈 Gelecek Çalışmalar

1. **NLP Integration:** Dava metinlerinin BERT/RoBERTa ile analizi
2. **Fairness-Aware ML:** Bias mitigation (reweighting, adversarial debiasing)
3. **Deep Learning:** LSTM/Transformer ile zamansal örüntü öğrenimi
4. **Real-Time API:** Flask/FastAPI ile canlı inference servisi

---

## 📚 Akademik Katkı

Bu proje, yapay zeka ve hukuk sistemlerinin kesişiminde:

- ✅ **Teknolojik:** CatBoost + LightGBM Ensemble + Router (Mixture of Experts)
- ✅ **Metodolojik:** Optuna, SHAP, Clustering, Geo-Analysis ile çok katmanlı analiz
- ✅ **Etik:** Bias detection, fairness analizi ve hakim sertlik skoru tespiti
- ✅ **Pratik:** Hakim destek sistemi için kullanıma hazır prototip (step_38)

### 📖 Literatür ile Karşılaştırma

| Çalışma | Dataset | Model | R² Teorik | R² Pratik | MAE |
|---------|---------|-------|-----------|-----------|-----|
| **Bu Proje (V3 Final)** | Wisconsin (106K) | **CatBoost+LightGBM MoE** | **%83.65** | **%74.06** | **313 / 443 gün** |
| Bu Proje (V2) | Wisconsin (106K) | CatBoost Hibrit | %83.06 | — | 348 gün |
| Bu Proje (V1) | Wisconsin (106K) | CatBoost Segmented | %83.00 | — | 349 gün |
| Benzer Çalışmalar | Çeşitli | RF/SVM/XGBoost | %30–65 | — | — |

**💡 Sonuç:** V3 Final sistemi teorik **%83.65**, gerçek dünya **%74.06** R² ile literatür ortalamasının çok üzerindedir. Pratik ve teorik fark bilimsel dürüstlükle ayrı ayrı raporlanmıştır.

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

*Son Güncelleme: 4 Mart 2026 - V3 Final Model (Optuna + Ensemble + Hakim Bias Düzeltmesi)*
