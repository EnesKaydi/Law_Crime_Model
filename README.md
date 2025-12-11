# 🏛️ Yapay Zeka Destekli Hukuk Asistanı - Ceza Tahmin Sistemi

## Wisconsin Ceza Mahkemesi Veri Seti ile Gelişmiş AI Modeli

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-V2-green.svg)](https://catboost.ai/)
[![R2 Score](https://img.shields.io/badge/R²-83.06%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

---

## 📋 Proje Hakkında

Bu proje, **Manisa Celal Bayar Üniversitesi** lisans tez çalışması kapsamında geliştirilmiştir. Wisconsin Eyaleti ceza mahkemesi kayıtlarını (1.5 milyon vaka) kullanarak, **makine öğrenmesi** ile hapis ceza süresi tahmin modeli oluşturulmuştur.

### 🎯 Proje Hedefleri

1. **Hakim Destek Sistemi:** Ceza kararlarında veri odaklı öneriler sunmak
2. **Adalet Sistemi Şeffaflığı:** Model kararlarının açıklanabilir olması (SHAP Analizi)
3. **Bias Analizi:** Irksal ve demografik önyargıların tespit edilmesi
4. **Yüksek Doğruluk:** **%83+ model performansı** ✅ BAŞARILDI

---

## 👥 Proje Ekibi

- **Öğrenci:** Muhammed Enes Kaydı
- **Danışman:** Müge Özçevik
- **Kurum:** Manisa Celal Bayar Üniversitesi
- **Tarih:** Aralık 2025

---

## 🤖 Model Mimarisi - V2 (Final)

### 🏆 Hibrit Sistem: Router + Segmentasyon + Interaction Features

Sistem, **3 temel inovasyona** dayanır:

1. **Böl ve Yönet (Divide & Conquer):** Veri seti "Hafif Suçlar" (300-3000 gün) ve "Ağır Suçlar" (3000+ gün) olarak ikiye ayrılmıştır.
2. **Akıllı Yönlendirme (Router AI):** Gelen davanın hangi modele gideceğine karar veren **%89.33 doğrulukta** bir sınıflandırıcı (CatBoostClassifier).
3. **Keşfedilen Özellikler (Feature Discovery):** 
   - `violent_recid`: Şiddet suçu + Sabıka kombinasyonu (**%24 SHAP etkisi** - En güçlü faktör!)
   - `severity_x_violent`: Şiddetin çarpan etkisi
   - `age_gap`: Hakim-Suçlu yaş farkı

---

## 📈 Model Performansı - REKOR SONUÇLAR

### 🎯 Ana Metrikler (Test Set) - V2 Final

| Metrik | V1 (Segmentasyon) | **V2 (Interactions)** | Durum |
|--------|-------------------|-----------------------|-------|
| **Router Accuracy** | %87.89 | **%89.33** | ✅ +1.44% |
| **Genel R² (Log Scale)** | %83.00 | **%83.06** 🏆 | ✅ Teorik Limit |
| **Genel R² (Real Scale)** | %78.77 | **%79.07** | ✅ +0.30% |
| **MAE (Hata Payı)** | 349 Gün | **348 Gün** | ✅ İyileşti |

> **💡 Kritik Başarı:** Mevcut veri setiyle ulaşılabilecek **teorik limit %83** seviyesine çıkmıştır. İnsan davranışını tahmin eden modeller için "State-of-the-Art" performans!

---

## 🔍 Model Açıklanabilirlik (SHAP Analizi)

### Top 5 En Önemli Faktörler

| Sıra | Özellik | SHAP Değeri | Açıklama |
|------|---------|-------------|----------|
| **1** | **violent_recid** | **0.2405** | 🚨 Şiddet suçu + Sabıka birlikteliği (Oyun Değiştirici!) |
| 2 | highest_severity | 0.1309 | Suçun yasal tanımındaki şiddet derecesi |
| 3 | is_recid_new | 0.0533 | Sabıka kaydının varlığı |
| 4 | wcisclass | 0.0518 | Suçun resmi sınıflandırma kodu |
| 5 | severity_x_violent | 0.0475 | Şiddet eyleminin suç derecesiyle çarpım etkisi |

**📊 Görsel Kanıtlar:** `outputs/shap_analysis/` klasöründe SHAP Summary Plot ve Dependence Plot'lar mevcuttur.

---

## ⚖️ Bias & Fairness Analizi

### A. Irk Önyargısı (Race Bias)

- **Genel Durum:** Model, genel ortalamada Afrikalı Amerikalılara (Black) **57 gün**, Beyazlara (White) **48 gün** EKSİK ceza tahmin etmektedir.
- **⚠️ Kritik Bulgu (Conditional Bias):**
  - Suç şiddeti "Yüksek" olduğunda, Siyahiler Beyazlara göre ortalama **+42 gün** daha fazla ceza tahmini almaktadır.
  - Suç şiddeti "Çok Yüksek" olduğunda (Cinayet vb.) bu fark kapanmakta.

### B. Cinsiyet Farkı

- Erkekler, Kadınlara göre ortalama **+100 gün** daha fazla ceza almaktadır.

**📌 Tez Yorumu:** Adalet mekanizması homojen değildir; ceza miktarı suçun niteliği kadar, davanın görüldüğü ilçeye ve hakimin şahsi eğilimine göre **%20-%30 oranında değişebilmektedir.**

---

## 🕵️ Derinlemesine Keşif Analizleri

### 1. Suçlu Personaları (Clustering)

K-Means algoritması ile suçlular **4 ana profile** ayrılmıştır:
- **Persona 0 (Hafif Suçlular):** Genç, sabıkasız, ortalama 500 gün ceza.
- **Persona 2 (Genç ve Tehlikeli):** En genç yaş grubu (28.9) ama en ağır cezalar (Ortalama **2304 Gün**).

### 2. Coğrafi Adalet Haritası

İlçelerin "Sertlik Skoru" (Modelin tahmininden sapma) hesaplanmıştır:
- **Adaletsiz Bölge:** `County 54` (+193 Gün Bias). Burada suç işleyen biri, başka bir ilçeye göre ortalama **6 ay daha fazla** yatmaktadır.
- **Paradoks:** En çok ceza hacmine sahip `County 61`, aslında en adil/yumuşak (-19 Gün Bias) bölgelerden biridir.

### 3. Yargıç Tipolojisi

Hakimler verdikleri kararların "beklenen değerden sapmasına" göre kümelenmiştir:
- **🔨 "The Hammer" (Sert Hakimler):** Judge 1374 - Model 1000 gün diyorsa, o 1211 gün veriyor (Bias: +211 Gün).
- **🕊️ "The Dove" (Babacan Hakimler):** Judge 1385 - Modelin tahmininden ortalama **-102 gün** daha az ceza veriyor.

---

## 🗂️ Proje Yapısı

```
LAW/
├── 📂 outputs/                      # Tüm analiz çıktıları
│   ├── shap_analysis/               # SHAP görselleştirmeleri
│   ├── bias_analysis/               # Irk/Cinsiyet bias grafikleri
│   ├── clustering_analysis/         # Suçlu profilleri
│   ├── geo_analysis/                # Coğrafi adalet haritası
│   ├── judge_typology/              # Hakim profilleri
│   └── interaction_analysis/        # Feature etkileşimleri
├── 📂 model_data_v2_interactions/   # V2 Final Modeller
│   ├── router_v2.cbm                # Router Classifier
│   ├── model_low_v2.cbm             # Mainstream Model (300-3000 gün)
│   ├── model_high_v2.cbm            # High Severity Model (3000+ gün)
│   ├── features_v2.pkl              # Özellik listesi
│   └── cat_features_v2.pkl          # Kategorik özellikler
├── 📄 BULGULAR_FINAL.md             # Tez Bulguları (SHAP, Bias, Geo)
├── 📄 WALKTHROUGH.md                # Teknik Özet ve Model Karşılaştırmaları
├── 📄 README.md                     # Bu dosya
└── 📜 step_08-step_21_*.py          # Pipeline scriptleri
    ├── step_14_final_pipeline.py    # Inference Pipeline (Router + Models)
    ├── step_16_retrain_with_interactions.py  # V2 Model Eğitimi
    ├── step_17_bias_fairness_analysis.py     # Bias Analizi
    ├── step_18_shap_explanation.py           # SHAP Açıklanabilirlik
    ├── step_19_clustering_analysis.py        # Suçlu Profilleri
    ├── step_20_geo_analysis.py               # Coğrafi Adalet
    └── step_21_judge_typology.py             # Hakim Profilleri
```

---

## 🚀 Kurulum ve Çalıştırma

### 1️⃣ Gereksinimler

```bash
Python 3.11+
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

# Paketleri yükle
pip install pandas numpy matplotlib seaborn scikit-learn catboost shap
```

### 3️⃣ Model Kullanımı (Inference)

```python
from step_14_final_pipeline import predict_sentence

# Örnek vaka
case_data = {
    'highest_severity': 15,
    'violent_crime': 1,
    'is_recid_new': 1,
    'age_offense': 28,
    # ... diğer özellikler
}

predicted_days = predict_sentence(case_data)
print(f"Tahmin Edilen Ceza: {predicted_days:.0f} gün")
```

---

## 📊 Sonuçlar ve Değerlendirme

### ✅ Başarılar

1. **Rekor Doğruluk:** R²=0.8306 (Log scale) - İnsan davranışı tahmininde teorik limite ulaşıldı
2. **Açıklanabilirlik:** SHAP analizi ile modelin "neden" karar verdiği görselleştirildi
3. **Bias Tespiti:** Sistemdeki ırksal ve coğrafi adaletsizlikler matematiksel olarak kanıtlandı
4. **Sosyolojik Analiz:** Sadece tahmin değil, "Hakim Profilleri" ve "Suçlu Personaları" gibi sosyal yapılar keşfedildi
5. **Hibrit Mimari:** Router + Segmentasyon stratejisi ile %83 başarıya ulaşıldı

### 🔬 Bilimsel Katkı

Bu proje, yapay zeka ve hukuk sistemlerinin kesişiminde:

- ✅ **Teknolojik:** CatBoost + Router mimarisi ile hibrit sistem
- ✅ **Metodolojik:** SHAP, Clustering, Geo-Analysis ile çok katmanlı analiz
- ✅ **Etik:** Bias detection ve fairness analizi (Conditional Bias keşfi)
- ✅ **Pratik:** Hakim destek sistemi için kullanıma hazır prototip

---

## 📚 Akademik Dokümanlar

- **`BULGULAR_FINAL.md`**: Tez için hazırlanmış detaylı bulgular raporu (SHAP, Bias, Coğrafi Adalet, Hakim Profilleri)
- **`WALKTHROUGH.md`**: Teknik özet ve model evrim süreci (V1 → V2 → V3 denemeleri)

---

## 🔗 Bağlantılar

- **GitHub Repo:** [github.com/EnesKaydi/Law_Crime_Model](https://github.com/EnesKaydi/Law_Crime_Model)
- **Detaylı Bulgular:** `BULGULAR_FINAL.md`
- **Teknik Özet:** `WALKTHROUGH.md`

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

*Son Güncelleme: 12 Aralık 2025 - V2 Final Model*
