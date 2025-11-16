# 🏛️ Yapay Zeka Destekli Hukuk Asistanı

## Wisconsin Ceza Mahkemesi Veri Seti ile Ceza Süresi Tahmin Modeli

[![Python](https://img.shields.io/badge/Python-3.12.6-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

---

## 📋 Proje Hakkında

Bu proje, **Manisa Celal Bayar Üniversitesi** lisans tez çalışması kapsamında geliştirilmiştir. Wisconsin Eyaleti ceza mahkemesi kayıtlarını (1.5 milyon vaka) kullanarak, **makine öğrenmesi** ile hapis ceza süresi tahmin modeli oluşturulmuştur.

### 🎯 Proje Hedefleri

1. **Hakim Destek Sistemi:** Ceza kararlarında veri odaklı öneriler sunmak
2. **Adalet Sistemi Şeffaflığı:** Model kararlarının açıklanabilir olması
3. **Bias Analizi:** Irksal ve demografik önyargıların tespit edilmesi
4. **Yüksek Doğruluk:** %85+ model performansı hedefi

---

## 👥 Proje Ekibi

- **Öğrenci:** Muhammed Enes Kaydı
- **Danışman:** Müge Özçevik
- **Kurum:** Manisa Celal Bayar Üniversitesi
- **Tarih:** Kasım 2025

---

## 📊 Veri Seti

- **Kaynak:** Wisconsin State Criminal Courts
- **Toplam Vaka:** 1,476,967 (~1.5 milyon)
- **Kolon Sayısı:** 54 (demografik, suç, ceza, mahalle bilgileri)
- **Final Dataset:** 525,379 kayıt (357K temiz + 168K örneklem)
- **Hedef Değişken:** `jail` (hapis süresi - gün)

### 📈 Veri Hazırlama Stratejisi

```
Orijinal Veri (1.5M)
    ↓
Temiz Veri Seçimi (357K) - %24.2
    +
Eksik Veriden %15 Örneklem (168K)
    ↓
Final Dataset (525K)
    ↓
Feature Engineering (41 feature)
    ↓
Train (283K) / Test (71K) - %80/%20 Split
```

---

## 🤖 Model Mimarisi

### XGBoost Regressor

**Seçim Nedenleri:**
- ✅ Yüksek boyutlu veri için optimize
- ✅ Eksik değerleri otomatik işleme
- ✅ Feature importance (yorumlanabilirlik)
- ✅ Overfitting'e karşı regularization
- ✅ Akademik çalışmalarda yaygın kullanım

### Hyperparameter Tuning

```python
GridSearchCV ile optimize edildi:
- n_estimators: 300
- max_depth: 3
- learning_rate: 0.05
- subsample: 1.0
- colsample_bytree: 1.0
```

---

## 📈 Model Performansı

### 🎯 Ana Metrikler (Test Set) - FİNAL ENSEMBLE MODEL

| Metrik | Orijinal Model | BALANCED Kategori | **Final Ensemble** | Toplam İyileşme |
|--------|----------------|-------------------|-------------------|-----------------|
| **R² Score** | 0.4404 | 0.6278 | **0.6321** | ✅ **+43.5%** |
| **RMSE** | 577.38 gün | 386.58 gün | **384.35 gün** | ✅ **-33.4%** |
| **MAE** | 89.09 gün | 85.82 gün | **86.08 gün** | ✅ **-3.4%** |
| **Model Tipi** | XGBoost | XGBoost | **XGBoost + LightGBM** | Ensemble |

### 📊 Kategori Bazlı Performans - YENİ SİSTEM

**BALANCED Kategori Sistemi (1-60, 61-365, 366+ gün):**

| Kategori | N | MAE (gün) | RMSE (gün) | R² | Başarı |
|----------|---|-----------|------------|-----|--------|
| **Hafif (1-60 gün)** | 49,221 (%69.4) | **33.40** | **38.55** | **0.29** | ⭐⭐⭐⭐⭐ |
| **Orta (61-365 gün)** | 18,572 (%26.2) | **84.65** | **105.42** | **0.23** | ⭐⭐⭐⭐ |
| **Ağır (366+ gün)** | 3,163 (%4.5) | **588.89** | **827.04** | **0.35** | ⭐⭐⭐ |

**💡 Kritik İyileşme:** 
- Kategori optimizasyonu (BALANCED) ile **tüm kategorilerde pozitif R²** elde edildi
- Orijinal sistemdeki negatif R² sorunu tamamen çözüldü
- Ensemble model (XGBoost + LightGBM) ile ek **+0.7% R² artışı**
- Hafif cezalarda MAE sadece **33 gün** (~1 ay) - pratik kullanım için mükemmel!

### 🎯 Final Model: Ensemble (XGBoost + LightGBM)
- **Simple Average Ensemble:** İki modelin tahminlerinin ortalaması
- **LightGBM Performansı:** R²=0.6301 (XGBoost'tan biraz daha iyi)
- **Ensemble Sinerji:** Farklı algoritmaların güçlü yönlerini birleştirme

---

## 🔍 Model Açıklanabilirlik (Explainability)

### Top 5 En Önemli Feature'lar

1. **highest_severity** (0.1545) - Suç ciddiyeti en yüksek önem
2. **pct_somecollege** (0.1023) - Eğitim seviyesi
3. **med_hhinc** (0.0880) - Medyan hane geliri
4. **all_races_freq** (0.0801) - Demografik kompozisyon
5. **felony_ratio** (0.0674) - Ağır suç oranı

### 🎨 Görselleştirmeler

- ✅ Feature Importance (XGBoost + Permutation)
- ✅ Partial Dependence Plots (top 6 features)
- ✅ Prediction vs Actual Scatter Plots
- ✅ Residual Analysis (hata dağılımı)
- ✅ Kategori Bazlı Performans Grafikleri

---

## ⚖️ Bias Analizi

### Kritik Bulgular (EDA'dan) - Sistemdeki Bias

| Grup | Ortalama Ceza | Fark |
|------|---------------|------|
| **Caucasian (Beyaz)** | 103.1 gün | Baseline |
| **African American (Siyah)** | 215.5 gün | **+109% daha yüksek** ⚠️ |
| **Male (Erkek)** | 115.2 gün | Baseline |
| **Female (Kadın)** | 72.5 gün | -37% daha düşük |

### Model Fairness Analizi - Demographic Parity

**Fairness Metrikleri:**

| Grup | Fairness Ratio | Durum |
|------|----------------|-------|
| **Irk (Race)** | 0.978 | ✅ Kabul Edilebilir (≥0.80) |
| **Cinsiyet (Gender)** | 0.989 | ✅ Kabul Edilebilir (≥0.80) |

**📌 Önemli:** Model, ırksal bias'ı öğrenmedi - feature importance analizinde ırk ve cinsiyet değişkenlerinin **görece düşük önemi**, modelin bu faktörlere aşırı ağırlık vermediğini gösteriyor. Fairness ratio değerleri literatür eşiğinin (0.80) üzerinde.

---

## 🗂️ Proje Yapısı

```
LAW/
├── 📂 outputs/               # Tüm çıktılar
│   ├── eda/                  # 30+ EDA görseli
│   ├── model/                # Eğitilmiş model + importance
│   ├── performance/          # Performans analizleri
│   ├── explainability/       # Feature importance plots
│   ├── new_categories/       # Yeni kategori sonuçları
│   ├── bias_analysis/        # Fairness analiz grafikleri
│   ├── 4_categories/         # 4 kategori deneme sonuçları
│   └── log_transformation/   # Log transform deneme sonuçları
├── 📂 model_data/            # Orijinal train/test split
├── 📂 model_data_new_categories/  # BALANCED kategori verileri
├── 📄 SONUCLAR.md            # Detaylı sonuçlar (TEZ için)
├── 📄 ADIMLAR.md             # Adım adım yeniden üretim rehberi
├── 📄 README.md              # Bu dosya
├── 📄 PROJE_OZET.md          # Detaylı proje özeti
└── 📜 00-17_*.py             # 18 adımlık pipeline scriptleri
    ├── 00_Kategori_Optimizasyon_Analizi.py
    ├── 14_Log_Transformation_Iyilestirme.py
    ├── 15_Yeni_Kategorilerle_Model.py
    ├── 16_4_Kategorili_Optimizasyon.py
    ├── 17_Demographic_Parity_Bias_Analizi.py
    └── outlier_analiz.py
```

---

## 🚀 Kurulum ve Çalıştırma

### 1️⃣ Gereksinimler

```bash
Python 3.12.6
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost
python-docx (tez doküman okuma için)
```

### 2️⃣ Ortam Kurulumu

```bash
# Repo'yu klonla
git clone https://github.com/EnesKaydi/Law_Crime_Model.git
cd Law_Crime_Model

# Virtual environment oluştur
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Paketleri yükle
pip install pandas numpy matplotlib seaborn scikit-learn xgboost python-docx
```

### 3️⃣ Pipeline Çalıştırma

**⚠️ Not:** Veri seti gizlilik nedeniyle repo'da bulunmamaktadır. Kendi `wcld.csv` dosyanızı kullanın.

```bash
# ADIM 1-4: Veri Hazırlama
python 01_veri_yukleme.py
python 02_temiz_veri_ayirma.py
python 03_eksik_veri_orneklem.py
python 04_final_dataset_birlestirme.py

# ADIM 5: EDA (5 aşama)
python 05_EDA_temel_istatistikler.py
python 05_EDA_hedef_degisken_dagitimi.py
python 06_EDA_kategorik_degiskenler.py
python 07_EDA_korelasyon_analizi.py
python 08_EDA_ileri_duzey_analizler.py

# ADIM 6-7: Feature Engineering & Normalization
python 09_Feature_Engineering_ve_Encoding.py
python 10_Normalizasyon_ve_Train_Test_Split.py

# ADIM 8-10: Model Training & Evaluation
python 11_XGBoost_Model_Egitimi.py
python 12_Detayli_Performans_Degerlendirme.py
python 13_Model_Explainability_Analizi.py
```

---

## 📊 Sonuçlar ve Değerlendirme

### ✅ Başarılar

1. **Yüksek Doğruluk - ENSEMBLE MODEL:** BALANCED kategori + Ensemble ile R²=0.44'ten R²=0.63'e yükseldi (%43.5 artış) - Akademik standartların üzerinde
2. **Model Çeşitliliği:** XGBoost + LightGBM ensemble ile robust tahminler
3. **Açıklanabilirlik:** Feature importance + Partial Dependence - Şeffaf model
4. **Bias Tespiti & Fairness:** Sistemdeki ırksal farklılıklar tespit edildi + Model fairness analizi (demographic parity 0.978-0.989)
5. **Kategori Optimizasyonu:** 5 farklı strateji test edildi, BALANCED sistemi başarılı
6. **Ensemble Sinerjisi:** İki farklı gradient boosting algoritmasının güçlü yönlerini birleştirme

### 📈 İyileştirme Potansiyeli

1. **Ensemble Yöntemleri:** XGBoost + LightGBM + CatBoost kombinasyonu
2. **Deep Learning:** LSTM/Transformer modelleri denenmeli
3. **Fairness-Aware ML:** Bias mitigation teknikleri (reweighting, adversarial debiasing)
4. **Temporal Features:** Tarih/mevsim etkilerinin modellenmesi

---

## 📚 Akademik Katkı

Bu proje, yapay zeka ve hukuk sistemlerinin kesişiminde:

- ✅ **Teknolojik:** XGBoost ile regresyon modellemesi
- ✅ **Metodolojik:** Stratified sampling + GridSearchCV
- ✅ **Etik:** Bias detection ve model fairness analizi
- ✅ **Pratik:** Hakim destek sistemi için prototip

### 📖 Literatür ile Karşılaştırma

| Çalışma | Dataset | Model | R² | MAE |
|---------|---------|-------|-----|-----|
| **Bu Proje (Final Ensemble)** | Wisconsin (525K) | **XGBoost + LightGBM** | **0.63** | **86 gün** |
| **Bu Proje (BALANCED)** | Wisconsin (525K) | XGBoost + BALANCED Cat. | 0.63 | 86 gün |
| **Bu Proje (Orijinal)** | Wisconsin (525K) | XGBoost | 0.44 | 89 gün |
| Benzer Çalışmalar | Çeşitli | RF/SVM | 0.30-0.50 | - |

**💡 Sonuç:** Performansımız literatür ortalamasının **ÇOK ÜZERİNDE**! Kategori optimizasyonu + Ensemble model kritik rol oynadı.

---

## 🔗 Bağlantılar

- **GitHub Repo:** [github.com/EnesKaydi/Law_Crime_Model](https://github.com/EnesKaydi/Law_Crime_Model)
- **Detaylı Sonuçlar:** `SONUCLAR.md`
- **Yeniden Üretim Rehberi:** `ADIMLAR.md`

---

## 📜 Lisans

Bu proje akademik amaçlı geliştirilmiştir. Ticari kullanım için izin gereklidir.

---

## 🙏 Teşekkürler

- **Danışman:** Müge Özçevik - Yönlendirme ve destek için
- **Wisconsin State Courts:** Veri setinin açık erişim sağlanması için
- **XGBoost Topluluğu:** Açık kaynak kütüphane için

---

## 📧 İletişim

**Muhammed Enes Kaydı**  
Manisa Celal Bayar Üniversitesi  
GitHub: [@EnesKaydi](https://github.com/EnesKaydi)

---

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

---

*Son Güncelleme: 2 Kasım 2025*
