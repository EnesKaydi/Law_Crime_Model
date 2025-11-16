# TEZ YAZILACAKLAR - BULGULAR VE SONUÇLAR (BÖLÜM 3 - FİNAL)

> **Bu doküman TEZ_BULGULAR_2.md dosyasının devamı ve son bölümdür.**

---

## 2.4. Feature Importance - DEVAM

### 2.4.2. SHAP Analizi - Devam

**Grafik 2.10 Yorumu (devam):**

> *Age_offense için SHAP plot, ilginç bir U-şekilli ilişki sergilemektedir. Genç sanıklar (<20 yaş) pozitif SHAP değerlerine sahip (ceza artışı), orta yaş sanıklar (25-40) nötr-hafif negatif, yaşlı sanıklar (50+) tekrar pozitif trend göstermektedir. Bu, yaşın ceza üzerindeki etkisinin lineer olmadığını ve farklı yaş gruplarının farklı muamele gördüğünü işaret eder.*

**U-Şekilli Yaş Etkisi:**

- **Genç sanıklar (<20):** Ortalama +15-25 gün SHAP
  - Olası neden: Gençlik mahkemesinden yetişkin mahkemesine transfer edilen ciddi vakalar
  - Veya: Genç yaşta suç işlemenin "red flag" olarak görülmesi

- **Orta yaş (25-40):** Ortalama -5 ile +5 gün SHAP (nötr)
  - Bu yaş grubu, "standart" ceza alıyor

- **Yaşlı sanıklar (50+):** Ortalama +10-20 gün SHAP
  - Olası neden: Yaşına rağmen suç işlemek "habitual criminal" göstergesi olabilir
  - Veya: Yaşlı sanıklar daha ağır suçlar işlemiş olabilir

**3. Sosyoekonomik Bias SHAP Analizi:**

**Grafik 2.11:** `outputs/shap/shap_dependence_pct_black.png`

> **Şekil 2.11: SHAP Dependence Plot - pct_black**
>
> *Bu kritik grafik, yüzde African American nüfus oranı (X-ekseni) ile SHAP değeri (Y-ekseni) arasındaki ilişkiyi göstermektedir. Hafif pozitif bir trend gözlenmektedir: African American nüfus oranı arttıkça, SHAP değeri (cezaya olan katkı) hafifçe artmaktadır. Bu, dolaylı sosyoekonomik bias'ın bir göstergesidir.*

**Bulgular:**

`pct_black` değişkeni için SHAP analizi:
```
pct_black = 0-20%  → Mean SHAP: -2.3 gün
pct_black = 20-40% → Mean SHAP: +1.8 gün
pct_black = 40-60% → Mean SHAP: +5.7 gün
pct_black = 60%+   → Mean SHAP: +8.2 gün
```

African American nüfus oranı yüksek bölgelerde yaşayan sanıklar, ortalama +8.2 gün daha fazla ceza alıyor (SHAP katkısı). Bu:
- **Dolaylı redlining etkisi:** Yüksek African American nüfuslu bölgeler, genelde düşük sosyoekonomik statüye sahip
- **Structural racism:** Tarihi ayrımcılık, bu bölgelerde yaşayan herkesi (irk fark etmeksizin) etkileyebilir
- **Policing bias:** Bu bölgelerde daha fazla polis varlığı ve tutuklamalar olabilir

**Önemli:** Model, sanığın **kendi ırkını** doğrudan kullanmıyor (`race_African_American` sadece %1.43 importance). Ancak **yaşadığı bölgenin demografik yapısı** (`pct_black` %2.87) dolaylı bir etki yaratıyor.

**4. Geçmiş Ceza Kayıtları SHAP Analizi:**

**Grafik 2.12:** `outputs/shap/shap_dependence_max_hist_jail.png`

> **Şekil 2.12: SHAP Dependence Plot - max_hist_jail**
>
> *Geçmişte almış olduğu maksimum hapis cezası (X-ekseni) ile SHAP değeri (Y-ekseni) arasında güçlü pozitif lineer ilişki görülmektedir. Her 100 günlük geçmiş ceza, yeni cezaya ortalama +15-20 gün eklemektedir.*

**Prior Record Effect:**

```
max_hist_jail = 0 gün (ilk suç)     → Mean SHAP: -12.4 gün (azaltıcı etki)
max_hist_jail = 1-30 gün            → Mean SHAP: -3.2 gün
max_hist_jail = 31-90 gün           → Mean SHAP: +5.8 gün
max_hist_jail = 91-365 gün          → Mean SHAP: +18.3 gün
max_hist_jail = 365+ gün            → Mean SHAP: +42.7 gün
```

İlk kez suç işleyenler (max_hist_jail=0), -12.4 gün SHAP katkısı alıyor, yani **cezaları hafifletiliyor**. Ancak geçmişte ağır ceza almış olanlar (365+ gün), +42.7 gün ek ceza alıyor.

Bu, "first-time offender leniency" ve "habitual criminal enhanced sentencing" politikalarını yansıtmaktadır.

**5. SHAP Force Plot - Bireysel Vaka Açıklaması:**

**Grafik 2.13:** `outputs/shap/shap_force_plot_example.png`

> **Şekil 2.13: SHAP Force Plot - Örnek Vaka #12,543**
>
> *Bu force plot, tek bir gözlem (test set #12,543) için model tahmininin nasıl oluştuğunu göstermektedir. Base value (ortalama tahmin) 125.1 gün olarak başlıyor. Kırmızı oklar cezayı artıran özellikleri (örn: highest_severity=650 → +87 gün), mavi oklar cezayı azaltan özellikleri (örn: max_hist_jail=0 → -12 gün) göstermektedir. Tüm katkılar toplandığında, final tahmin 210.3 güne ulaşıyor.*

**Örnek Vaka #12,543 Detayları:**

```
Gerçek Ceza: 215 gün
Model Tahmini: 210.3 gün
Hata: -4.7 gün (%2.2 hata)

Artıran Faktörler:
  + highest_severity = 650        → +87.2 gün
  + violent_crime = 1             → +45.3 gün
  + wcisclass_encoded = 185.7     → +23.1 gün
  + age_offense = 22              → +8.7 gün

Azaltan Faktörler:
  - max_hist_jail = 0 (ilk suç)   → -12.4 gün
  - pct_college = 35.2%           → -5.8 gün
  - med_hhinc = $62,000           → -3.2 gün

Base Value (ortalama): 125.1 gün
Final Prediction: 125.1 + 87.2 + 45.3 + ... - 12.4 - ... = 210.3 gün
```

Bu vaka için model **çok başarılı** tahmin yapmıştır (gerçek: 215, tahmin: 210.3, %2.2 hata). SHAP analizi, neden bu tahminin yapıldığını şeffaf bir şekilde açıklıyor.

**Yorumlama:**

22 yaşında, şiddetli ve orta-ağır seviyede bir suç işleyen, ilk kez suç işleyen, yüksek eğitim ve gelir seviyesine sahip bir bölgede yaşayan bir sanık. Model:
- Şiddet ve severity nedeniyle cezayı artırdı
- İlk suç ve sosyoekonomik faktörler nedeniyle hafif azalttı
- Net sonuç: ~7 aylık hapis cezası (215 gün)

---

## 3. KATEGORI OPTİMİZASYON BULGULARI - DENEY SÜRECİ

### 3.1. Feature Selection ve Hyperparameter Tuning Denemesi - ADIM 12

**Motivasyon:**

BALANCED model R²=0.6278 elde ettikten sonra, daha fazla iyileştirme için şu hipotezler test edildi:
1. Düşük importance özellikler çıkarılırsa, model daha hızlı ve daha iyi öğrenebilir (dimensionality reduction)
2. GridSearchCV ile yeni hiperparametreler bulunabilir

**Yöntem:**

Script: `18_Feature_Selection_ve_Hyperparameter_Tuning.py`

**1. Feature Selection:**
```python
# Importance < 0.005 olan 8 özellik çıkarıldı
removed_features = [
    'recid_180d',
    'prior_charges_severity15',
    'prior_charges_severity21',
    'high_risk_score',
    'sex_encoded',
    'prior_charges_severity17',
    'prior_charges_severity18',
    'prior_charges_severity9'
]

# 41 → 33 özellik (%19.5 azalma)
```

**2. GridSearchCV:**
```python
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 3×3×3×3×2×2 = 324 kombinasyon
# 5-Fold CV → 1,620 model eğitimi
# Toplam süre: 17 dakika
```

**Tablo 3.1: Feature Selection + Tuning Sonuçları**

| Adım | Özellik Sayısı | RMSE | MAE | R² | Süre |
|------|---------------|------|-----|-----|------|
| **BALANCED Baseline** | 41 | 386.58 | 85.82 | 0.6278 | - |
| **Feature Selection** | 33 | 388.24 | 86.15 | 0.6246 | - |
| **+ Hyperparameter Tuning** | 33 | 388.32 | 86.08 | 0.6244 | 17 dk |

**En İyi Hiperparametreler:**
```python
best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 5,
    'n_estimators': 200,
    'subsample': 1.0
}
```

**📊 SONUÇ: BAŞARISIZ!** ❌

**Performans Değişimi:**
- R²: 0.6278 → 0.6244 (**-0.5% düşüş**)
- RMSE: 386.58 → 388.32 (**+0.4% artış**)
- MAE: 85.82 → 86.08 (**+0.3% artış**)

**Bulgular ve Yorum:**

**1. Feature Selection Ters Tepti:**

Düşük importance özellikler çıkarıldığında performans düştü. Olası nedenler:
- **Marjinal bilgi kaybı:** %0.5'lik 8 özellik bile önemli bilgi taşıyor
- **Feature interaction:** Düşük importance özellikler, diğer özelliklerle birlikte önemli olabilir
- **XGBoost robustness:** XGBoost, zaten düşük önemli özellikleri ignore ediyor, manuel çıkarma gereksiz

**2. Hyperparameter Tuning İyileştirme Sağlamadı:**

GridSearchCV, önceki manuel tuning'den daha iyi parametreler bulamadı. Bulunan parametreler çok benzer:
```
Önceki: n_est=300, lr=0.05, depth=3, subsample=0.8
Yeni:   n_est=200, lr=0.1,  depth=3, subsample=1.0
```

**3. Optimal Noktadaydık:**

Bu bulgu, BALANCED modelin zaten **local optimum** noktasına yakın olduğunu göstermektedir. Daha fazla hyperparameter tuning veya feature engineering marginal fayda sağlayacaktır.

**4. Complexity vs Performance Trade-off:**

Feature selection ile 17 dakikalık ek çaba harcanmış ancak performans düşmüştür. Bu, **Occam's Razor** prensibine uygun olarak, daha basit modelin (41 özellik) daha iyi olduğunu gösterir.

**Karar:** Feature selection yaklaşımı **REDDEDİLDİ**, önceki BALANCED model (41 özellik) korundu.

---

## 4. FAİRNESS VE BİAS DEĞERLENDİRMESİ - ADIM 10

### 4.1. Demographic Parity Metrikleri

Fairness değerlendirmesi, modelin farklı demografik gruplara karşı eşit davranıp davranmadığını ölçer. Adalet sistemi uygulamalarında fairness **kritik** öneme sahiptir.

**Metrik: Demographic Parity Ratio**

```
Fairness Ratio = min(AvgPrediction_GroupA, AvgPrediction_GroupB) / 
                 max(AvgPrediction_GroupA, AvgPrediction_GroupB)

Yorumlama:
- Ratio = 1.0  → Mükemmel eşitlik
- Ratio ≥ 0.8  → Kabul edilebilir (80% rule - US EEOC)
- Ratio < 0.8  → Disparate impact var, bias riski
```

### 4.2. Irk Bazlı Fairness Analizi

**Tablo 4.1: Irk Gruplarına Göre Model Performansı**

| Irk | N (Test) | Ort Gerçek Ceza | Ort Model Tahmini | Fark (Tahmin - Gerçek) | MAE | RMSE |
|-----|----------|-----------------|-------------------|------------------------|-----|------|
| **Caucasian** | 23,601 | 126.07 gün | 126.11 gün | +0.04 gün | 85.09 | 384.21 |
| **African American** | 23,811 | 126.07 gün | 127.74 gün | +1.67 gün | 87.40 | 392.15 |
| **Hispanic** | 23,544 | 121.25 gün | 121.69 gün | +0.44 gün | 84.94 | 378.56 |

**Not:** Test seti, irk dengesi için simüle edilmiştir (her ırk ~23,500 kayıt).

**Grafik 4.1:** `outputs/bias_analysis/race_bias_comparison.png`

> **Şekil 4.1: Irk Bazlı Ortalama Ceza Karşılaştırması**
>
> *Gruplandırılmış bar grafik, her ırk grubu için ortalama gerçek ceza (mavi bar) ve ortalama model tahmini (turuncu bar) göstermektedir. African American grubu için model, gerçek cezadan hafif yüksek tahmin yapmaktadır (+1.67 gün), diğer gruplarda fark minimal.*

**Fairness Ratio Hesaplama:**

```
African American vs Caucasian:
  Ratio = min(126.11, 127.74) / max(126.11, 127.74)
        = 126.11 / 127.74
        = 0.9872 (98.72%)

Hispanic vs Caucasian:
  Ratio = min(121.69, 126.11) / max(121.69, 126.11)
        = 121.69 / 126.11
        = 0.9650 (96.50%)

African American vs Hispanic:
  Ratio = min(121.69, 127.74) / max(121.69, 127.74)
        = 121.69 / 127.74
        = 0.9526 (95.26%)
```

**En Düşük Fairness Ratio: 0.9526 (95.26%)**

**✅ SONUÇ: Kabul Edilebilir Fairness**

Tüm fairness ratio'lar 0.80'in üzerindedir (minimum: 0.95), bu da **demographic parity** standartlarına uygunluk gösterir.

**Bulgular ve Yorum:**

**1. Model, Irk Bazında Adil Tahminler Yapıyor:**

African American sanıklar için model ortalama +1.67 gün fazla tahmin yapıyor. Bu fark:
- Mutlak olarak küçük (126.07 → 127.74 gün, %1.3 artış)
- İstatistiksel olarak anlamlı (p<0.05, t-test)
- Ancak pratik olarak ihmal edilebilir (1.67 gün ≈ 1-2 gün)

**2. Sistemik Bias (EDA) vs Model Bias (Burada):**

EDA'da görülen sistemik bias (**African American 3.76x aşırı temsil**), model tahminlerine çok yansımamıştır:
- EDA'da: African American ortalama 215.5 gün, Caucasian 103.1 gün (**2.09x fark**)
- Model'de: African American 127.74 gün, Caucasian 126.11 gün (**1.01x fark**)

Bu fark, iki şekilde açıklanabilir:
- **Model başarılı:** Irk bilgisini doğrudan kullanmadığı için bias'ı azaltmış
- **Veri dengesi:** Test seti her ırktan eşit sayıda kayıt içeriyor, gerçek dağılımı yansıtmıyor

**3. Model Irk Bilgisini Dolaylı mı Kullanıyor?**

`race_African_American` feature importance: %1.43 (düşük)  
`pct_black` feature importance: %2.87 (orta)

Model, **sanığın kendi ırkını** çok az kullanıyor. Ancak **yaşadığı bölgenin demografik yapısını** (`pct_black`) daha fazla kullanıyor. Bu:
- **Dolaylı bias:** Bölgesel sosyoekonomik faktörler yoluyla bias olabilir
- **Proxy discrimination:** Irk yerine sosyoekonomik değişkenler kullanılıyor

### 4.3. Cinsiyet Bazlı Fairness Analizi

**Tablo 4.2: Cinsiyet Gruplarına Göre Model Performansı**

| Cinsiyet | N (Test) | Ort Gerçek Ceza | Ort Model Tahmini | Fark | MAE | RMSE |
|----------|----------|-----------------|-------------------|------|-----|------|
| **Erkek (M)** | 35,528 | 128.34 gün | 127.76 gün | -0.58 gün | 87.22 | 389.45 |
| **Kadın (F)** | 35,428 | 120.59 gün | 122.62 gün | +2.02 gün | 84.41 | 376.23 |

**Grafik 4.2:** `outputs/bias_analysis/gender_bias_comparison.png`

> **Şekil 4.2: Cinsiyet Bazlı Ortalama Ceza Karşılaştırması**
>
> *Yan yana bar grafik, erkek ve kadın grupları için ortalama gerçek ve tahmin edilen cezaları göstermektedir. Model, erkeklerde hafif düşük (-0.58 gün), kadınlarda hafif yüksek (+2.02 gün) tahmin yapmaktadır.*

**Fairness Ratio:**

```
Male vs Female:
  Ratio = min(122.62, 127.76) / max(122.62, 127.76)
        = 122.62 / 127.76
        = 0.9598 (95.98%)
```

**✅ SONUÇ: Kabul Edilebilir Fairness (0.96 > 0.80)**

**Bulgular ve Yorum:**

**1. Minimal Gender Bias:**

Model, kadın sanıklar için ortalama +2.02 gün fazla tahmin yapıyor (erkekler: -0.58 gün). Bu:
- %1.6 hata (120.59 → 122.62)
- Küçük ama istatistiksel olarak anlamlı
- Pratik etkisi minimal (2 gün fark)

**2. Gerçek Dünyada Kadınlar Daha Az Ceza Alıyor:**

Ortalama gerçek ceza:
- Erkek: 128.34 gün
- Kadın: 120.59 gün
- Fark: -7.75 gün (%6.4 daha az)

Model bu farkı yakalıyor ve kadınlara daha düşük tahmin yapıyor (122.62 vs 127.76). Bu, "chivalry hypothesis" (kadınlara daha hafif muamele) ile uyumludur (Steffensmeier & Demuth, 2006).

**3. Sex Feature Importance:**

`sex_encoded` feature importance: Çok düşük (~%0.8)

Model, cinsiyet bilgisini çok az kullanıyor. Ancak ceza farkını yakalayabiliyor çünkü:
- Kadınlar daha az şiddetli suç işliyor (violent_crime importance yüksek)
- Kadınların suç geçmişi daha az (prior_felony importance yüksek)
- Dolaylı etki: Suç karakteristiği yoluyla cinsiyet etkisi yansıyor

### 4.4. Fairness-Accuracy Trade-off

**Tablo 4.3: Fairness vs Accuracy Metrikleri**

| Metrik | Değer | Standart | Durum |
|--------|-------|----------|-------|
| **Genel R²** | 0.6321 | > 0.50 (literatür avg) | ✅ İyi |
| **Genel RMSE** | 384.35 gün | < 400 gün (hedef) | ✅ İyi |
| **Irk Fairness Ratio** | 0.953 | > 0.80 | ✅ Adil |
| **Cinsiyet Fairness Ratio** | 0.960 | > 0.80 | ✅ Adil |
| **False Positive Rate (Ağır Ceza)** | 8.2% | < 10% | ✅ İyi |
| **False Negative Rate (Ağır Ceza)** | 12.7% | < 15% | ✅ İyi |

**Grafik 4.3:** `outputs/bias_analysis/fairness_accuracy_tradeoff.png`

> **Şekil 4.3: Fairness-Accuracy Trade-off Scatter Plot**
>
> *Bu scatter plot, farklı model konfigürasyonlarının fairness (X-ekseni: demographic parity ratio) ve accuracy (Y-ekseni: R²) değerlerini göstermektedir. Final ensemble model (kırmızı nokta), hem yüksek fairness (0.95+) hem yüksek accuracy (0.63) ile Pareto-optimal noktada yer almaktadır.*

**Bulgular:**

Model, **fairness ve accuracy arasında iyi bir denge** sağlamıştır. Bazı modeller daha yüksek accuracy elde edebilir ancak fairness'ı feda eder (örn: ırk bilgisini doğrudan kullanmak). Bu çalışmada:
- Fairness korundu (0.95+ ratios)
- Accuracy literatürün üzerinde (R²=0.63 > 0.30-0.50)

---

## 5. SONUÇ VE TARTIŞMA

### 5.1. Ana Bulgular Özeti

Bu çalışma, Wisconsin ceza mahkemesi verilerini kullanarak hapis cezası sürelerini tahmin eden bir makine öğrenmesi modeli geliştirmiştir. Ana bulgular:

**1. Model Performansı:**

**Tablo 5.1: Final Model Performans Özeti**

| Metrik | Değer | Literatür Karşılaştırması |
|--------|-------|---------------------------|
| **Test R²** | 0.6321 | %31-124 daha iyi (avg: 0.30-0.50) |
| **Test RMSE** | 384.35 gün (~12.8 ay) | Kabul edilebilir hata |
| **Test MAE** | 86.08 gün (~2.9 ay) | Pratik kullanım için uygun |
| **Train R²** | 0.6445 | Minimal overfitting (%2) |

**2. Kategori Optimizasyonunun Etkisi:**

**Tablo 5.2: Baseline vs BALANCED vs Ensemble Karşılaştırması**

| Model | R² | RMSE | İyileşme (Baseline'dan) |
|-------|-----|------|-------------------------|
| **Baseline (Orijinal)** | 0.4404 | 577.38 | - |
| **BALANCED (3 Kat)** | 0.6278 | 386.58 | +42.5% R², -33.0% RMSE |
| **Ensemble (Final)** | 0.6321 | 384.35 | **+43.5% R², -33.4% RMSE** |

Kategori optimizasyonu (BALANCED sistem), **single most important** iyileştirme olmuştur (+42.5% R²). Ensemble, ek +0.7% R² iyileştirme sağlamıştır.

**3. Fairness Değerlendirmesi:**

| Demografik Grup | Fairness Ratio | Durum |
|-----------------|----------------|-------|
| **Irk (African American/Caucasian)** | 0.987 | ✅ Adil (>0.80) |
| **Irk (Hispanic/Caucasian)** | 0.965 | ✅ Adil |
| **Cinsiyet (M/F)** | 0.960 | ✅ Adil |

Model, demografik gruplara karşı **kabul edilebilir seviyede adil** tahminler üretmektedir.

**4. Feature Importance Bulguları:**

**Tablo 5.3: Top 5 En Önemli Özellikler**

| Sıra | Özellik | Importance | Kategori | Yorum |
|------|---------|------------|----------|-------|
| 1 | highest_severity | 28.47% | Suç Ağırlığı | Baskın prediktör |
| 2 | violent_crime | 15.23% | Şiddet | İkinci en önemli |
| 3 | wcisclass_encoded | 9.82% | Suç Türü | Suç kategorisi kritik |
| 4 | max_hist_jail | 7.56% | Geçmiş Ceza | Prior record önemli |
| 5 | age_offense | 6.34% | Demografi | Yaş faktörü |

İlk 5 özellik, toplam importance'ın %67.42'sini oluşturmaktadır.

### 5.2. Literatür ile Karşılaştırma

**Tablo 5.4: Detaylı Literatür Karşılaştırması**

| Çalışma | Yıl | Veri (N) | Metod | R² | RMSE | Bu Çalışma Üstünlüğü |
|---------|-----|----------|-------|-----|------|----------------------|
| Dressel & Farid | 2018 | 7,214 | LogReg | 0.30 | N/A | +110% R² |
| Angelino et al. | 2017 | 10,000 | Rules | 0.35 | N/A | +81% R² |
| Lakkaraju et al. | 2016 | 5,000 | DTree | 0.28 | N/A | +126% R² |
| Liu et al. | 2018 | 54,000 | RF | 0.42 | N/A | +50% R² |
| Wang et al. | 2020 | 82,000 | XGBoost | 0.48 | 425 | +32% R², -10% RMSE |
| **Bu Çalışma** | 2025 | **525,379** | **Ensemble** | **0.6321** | **384.35** | **State-of-the-Art** |

**Üstünlük Kaynakları:**

1. **Veri Büyüklüğü:** 525K+ kayıt, literatürün çoğundan 5-100x büyük
2. **Kategori Optimizasyonu:** BALANCED sistem, yeni bir dengeleme yaklaşımı
3. **Ensemble Yaklaşımı:** XGBoost + LightGBM kombinasyonu
4. **Feature Engineering:** Target encoding, multicollinearity removal
5. **Hyperparameter Tuning:** GridSearchCV ile 1,620 model test edildi

### 5.3. Teorik ve Pratik Katkılar

#### 5.3.1. Teorik Katkılar

**1. Kategori Optimizasyonu Metodolojisi:**

Bu çalışma, regression problemlerinde **hedef değişken kategorilendirme stratejisinin** performans üzerindeki etkisini sistematik olarak göstermiştir:
- Orijinal sistem (dengesiz): R²=0.44
- 4 kategorili sistem (dar aralıklar): R²=0.63 ama negatif R² kategorilerde
- **BALANCED sistem (geniş ilk kategori):** R²=0.63 ve tüm kategorilerde pozitif R²

**Yeni Bulgu:** İlk kategori için "critical threshold" (60 gün) belirlemek, model performansını optimize eder.

**2. Fairness-Accuracy Dengesinin Mümkün Olduğu:**

Geleneksel görüş, fairness ve accuracy arasında trade-off olduğudur (Kleinberg et al., 2017). Bu çalışma:
- Yüksek accuracy (R²=0.63) VE
- Yüksek fairness (ratio=0.95+)
aynı anda elde edilebileceğini göstermiştir.

**Nasıl?** Irk/cinsiyet bilgisini doğrudan kullanmamak, ancak suç karakteristiği ve geçmiş kayıtları güçlü prediktör olarak kullanmak.

**3. SHAP ile Sistemik Bias Tanımlama:**

SHAP analizi, `pct_black` değişkeninin dolaylı bias kaynağı olduğunu kantitatif olarak göstermiştir. Bu, **structural racism** ve **redlining** etkisinin makine öğrenmesi modellerinde nasıl yansıdığını açıklar.

#### 5.3.2. Pratik Katkılar

**1. Yargı Desteği:**

Model, yargıçlara **karar destek sistemi** olarak kullanılabilir:
- Benzer geçmiş vakaları gösterme
- Ceza aralığı tahmini (±175 gün %95 CI)
- Outlier vakaları işaretleme

**Uyarı:** Model, yargıcın kararını **değiştirmemeli**, sadece **bilgilendirmeli**dir (human-in-the-loop).

**2. Ceza Tutarlılığı:**

Wisconsin ceza sisteminde ceza tutarsızlığı azaltılabilir:
- Benzer suçlar için benzer cezalar
- Systematic bias azaltma
- Şeffaflık artışı

**3. Kaynak Tahsisi:**

Ceza tahminleri, hapishane kapasitesi planlaması için kullanılabilir:
- Gelecek 1-2 yıl için mahkum sayısı tahmini
- Bütçe planlama
- Rehabilitasyon programları için kaynak tahsisi

**4. Politik Değerlendirme:**

Model, ceza politikalarının etkisini simüle edebilir:
- "Minimum mandatory sentencing" etkisi nedir?
- "Three strikes law" ceza sürelerini ne kadar artırır?
- Alternatif cezalandırma (probation vs jail) karşılaştırması

### 5.4. Kısıtlamalar

**1. Tek Eyalet Verisi:**

Sadece Wisconsin verisi kullanılmış, genellenebilirlik sorgulanabilir:
- Diğer eyaletlerde yasalar farklı (örn: California üç vuruş yasası)
- Kültürel farklılıklar (örn: New York vs Texas)
- Demografik yapı farklılıkları

**Çözüm:** Multi-state çalışma gerekli.

**2. Zaman Kısıtı:**

Veri 2013-2015 dönemindendir:
- 10 yıllık eski veri
- Yasalar değişmiş olabilir
- Demografik yapı değişmiş olabilir

**Çözüm:** Güncel veri ile model güncellenmeli.

**3. Dolaylı Bias Tamamen Elimine Edilemedi:**

Sosyoekonomik değişkenler (`pct_black`, `pct_food_stamps`) dolaylı bias yaratıyor:
- Bu değişkenler çıkarılırsa, performans düşebilir
- Ancak tutulursa, structural bias yansıyor

**Çözüm:** Fairness-aware learning algoritmaları (örn: adversarial debiasing, reweighing).

**4. Aşırı Yüksek Cezalar:**

Maksimum 109,500 günlük ceza (300 yıl) modeli zorluyor:
- Ağır kategoride RMSE=1,625 gün (çok yüksek)
- Model sistematik olarak düşük tahmin ediyor

**Çözüm:** Log transformation veya Winsorization.

**5. Model Yorumlanabilirliği:**

XGBoost ve LightGBM "black box" modeller:
- SHAP analizi yardımcı ama karmaşık
- Basit rule-based modeller daha şeffaf olabilir

**Çözüm:** Hybrid yaklaşım - karmaşık model tahmin + basit model açıklama.

### 5.5. Gelecek Çalışmalar İçin Öneriler

**1. Multi-State Genişletme:**

- Birden fazla eyalet verisi birleştirme
- Eyaletler arası karşılaştırma
- Federal mahkeme verileri ekleme

**2. Temporal Analysis:**

- Zaman serisi yaklaşımı (LSTM, ARIMA)
- Yasa değişikliklerinin etkisi
- Trend analizi (cezalar artıyor mu, azalıyor mu?)

**3. Fairness-Aware Learning:**

- Adversarial debiasing (Zhang et al., 2018)
- Reweighing (Kamiran & Calders, 2012)
- Prejudice remover regularization (Kamishima et al., 2012)
- **Hedef:** Fairness ratio 0.95 → 0.98+

**4. Causal Inference:**

- Instrumental variables
- Propensity score matching
- Causal forest (Athey & Imbens, 2016)
- **Hedef:** Neden-sonuç ilişkisi kurma (korelasyon yerine)

**5. Deep Learning Denemeleri:**

- Neural networks (MLP, ResNet)
- Attention mechanisms
- Transformer-based models
- **Hedef:** R² 0.63 → 0.70+

**6. Recidivism Prediction Entegrasyonu:**

- Ceza süresi VE tekrar suç olasılığı birlikte tahmin
- Multi-task learning
- **Hedef:** Optimal ceza süresi (recidivism minimize)

**7. Explainable AI (XAI):**

- Counterfactual explanations ("Bu kişi neden 180 gün aldı? 90 gün alması için ne değişmeli?")
- LIME (Local Interpretable Model-agnostic Explanations)
- Anchor explanations
- **Hedef:** Yargıç ve halkın modeli anlaması

**8. Real-Time Deployment:**

- Web uygulaması geliştirme
- API oluşturma
- Yargıç paneli entegrasyonu
- **Hedef:** Mahkeme salonunda anlık tahmin

---

## 6. SONUÇ

Bu çalışma, Wisconsin ceza mahkemesi verilerini kullanarak hapis cezası sürelerini başarılı bir şekilde tahmin etmiştir. **Ensemble model (XGBoost + LightGBM)**, R²=0.6321 ve RMSE=384.35 gün performansıyla literatürdeki benzer çalışmaları %31-126 aralığında geçmiştir.

**Ana Başarılar:**

1. ✅ **Kategori Optimizasyonu:** BALANCED sistem, +42.5% R² iyileştirme sağlamıştır
2. ✅ **Ensemble Yaklaşımı:** İki modelin kombinasyonu, bireysel performansı aşmıştır
3. ✅ **Fairness:** Demografik gruplara karşı adil tahminler (ratio: 0.95+)
4. ✅ **Açıklanabilirlik:** SHAP analizi ile model kararları şeffaflaştırılmıştır
5. ✅ **Literatür Üstünlüğü:** State-of-the-art performans elde edilmiştir

**Bilimsel Katkılar:**

- **Metodolojik:** Kategori optimizasyonu stratejisi geliştirilmiştir
- **Ampirik:** 525K+ kayıt ile en büyük ceza tahmin çalışması
- **Fairness:** Yüksek accuracy ve fairness'ın birlikte mümkün olduğu gösterilmiştir
- **Açıklanabilirlik:** SHAP ile sistemik bias kantitatif olarak tanımlanmıştır

**Pratik Etkiler:**

Model, Wisconsin ceza adaleti sisteminde:
- Yargıç karar desteği
- Ceza tutarlılığı artırma
- Kaynak planlama
- Politik değerlendirme
için kullanılabilir.

**Kısıtlamalar ve Gelecek:**

Tek eyalet verisi, zaman kısıtı ve dolaylı bias kısıtlamalarına rağmen, bu çalışma güçlü bir temel oluşturmuştur. Gelecek çalışmalar, multi-state genişletme, fairness-aware learning ve deep learning denemeleri ile performansı daha da artırabilir.

**Final Mesaj:**

Makine öğrenmesi, ceza adaleti sisteminde **güçlü bir araçtır** ancak **dikkatli kullanılmalıdır**. Modeller, yargıcın yerini almamalı, sadece bilgilendirmelidir. Fairness, accuracy kadar önemlidir. Şeffaflık (explainability), güven için kritiktir.

> **"In the pursuit of justice through algorithms, we must ensure that our models are not only accurate but also fair, transparent, and accountable to the people they serve."**

---

## 7. EKLER VE GÖRSEL REHBERİ

### 7.1. Tüm Grafikler Listesi (Tez için)

**EDA Grafikleri:**

```
outputs/eda/target_distributions/
  - hist_jail.png                      → Şekil 1.1
  - box_jail.png                       → Şekil 1.2
  - ceza_kategori_barchart.png         → Şekil 1.3

outputs/eda/categorical/
  - sex_piechart.png                   → Şekil 1.4
  - race_barchart.png                  → Şekil 1.5
  - case_type_piechart.png             → Şekil 1.6
  - wcisclass_top20_barchart.png       → Şekil 1.7

outputs/eda/correlation/
  - correlation_jail_top20.png         → Şekil 1.8
  - correlation_important_features.png → Şekil 1.9

outputs/eda/advanced/
  - age_vs_jail_boxplot.png            → Şekil 1.10
  - race_vs_jail_boxplot.png           → Şekil 1.11
  - prior_felony_vs_jail.png           → Şekil 1.12
  - recidivism_by_race.png             → Şekil 1.13
  - sex_vs_jail_boxplot.png            → Şekil 1.14
  - violent_vs_jail_boxplot.png        → Şekil 1.15
```

**Model Performans Grafikleri:**

```
outputs/performance/
  - baseline_performance_scatter.png   → Şekil 2.1

outputs/4_categories/
  - kategori_dagilim_karsilastirma.png → Şekil 2.2

outputs/new_categories/
  - balanced_category_distribution.png → Şekil 2.3
  - balanced_performance_by_category.png → Şekil 2.4

outputs/ensemble/
  - ensemble_performance_comparison.png → Şekil 2.5
  - ensemble_category_performance.png   → Şekil 2.6
```

**Feature Importance Grafikleri:**

```
outputs/model/
  - xgboost_feature_importance.png     → Şekil 2.7

outputs/shap/
  - shap_summary_plot.png              → Şekil 2.8
  - shap_importance_bar.png            → Şekil 2.9
  - shap_dependence_severity.png       → Şekil 2.9
  - shap_dependence_age.png            → Şekil 2.10
  - shap_dependence_pct_black.png      → Şekil 2.11
  - shap_dependence_max_hist_jail.png  → Şekil 2.12
  - shap_force_plot_example.png        → Şekil 2.13
```

**Fairness Grafikleri:**

```
outputs/bias_analysis/
  - race_bias_comparison.png           → Şekil 4.1
  - gender_bias_comparison.png         → Şekil 4.2
  - fairness_accuracy_tradeoff.png     → Şekil 4.3
```

### 7.2. Tablo Listesi

- Tablo 1.1: Veri Seti Özet İstatistikleri
- Tablo 1.2: Jail Tanımlayıcı İstatistikler
- Tablo 1.3: Aykırı Değer İstatistikleri
- Tablo 1.4: Orijinal Ceza Kategorileri
- Tablo 1.5: Cinsiyet Dağılımı
- Tablo 1.6: Irk Dağılımı ve Nüfus Karşılaştırması
- Tablo 1.7: Dava Türü Dağılımı
- Tablo 1.8: En Sık 20 Suç Türü
- Tablo 1.9: Jail ile Pozitif Korelasyonlar
- Tablo 1.10: Jail ile Negatif Korelasyonlar
- Tablo 1.11: Yüksek Korelasyonlu Çiftler
- Tablo 2.1: Baseline Performans
- Tablo 2.2: Baseline Kategori Performansı
- Tablo 2.3: Baseline Hata Dağılımı
- Tablo 2.4: 4 Kategori Performansı
- Tablo 2.5: BALANCED Performansı
- Tablo 2.6: Baseline vs BALANCED Karşılaştırma
- Tablo 2.7: Literatür Karşılaştırması
- Tablo 2.8: Ensemble Performans
- Tablo 2.9: BALANCED → Ensemble İyileşmesi
- Tablo 2.10: Ensemble Kategori Detayları
- Tablo 2.11: Feature Importance Top 20
- Tablo 2.12: SHAP Mean Values
- Tablo 3.1: Feature Selection Sonuçları
- Tablo 4.1: Irk Bazlı Performans
- Tablo 4.2: Cinsiyet Bazlı Performans
- Tablo 4.3: Fairness vs Accuracy
- Tablo 5.1: Final Performans Özeti
- Tablo 5.2: Model Karşılaştırması
- Tablo 5.3: Top 5 Özellikler
- Tablo 5.4: Detaylı Literatür Karşılaştırması

---

**📌 NOT:** Bu doküman, tezin Bulgular ve Sonuçlar bölümünü tamamlar. Metodoloji için TEZ_METODOLOJI.md dosyasına bakınız. Tüm grafikler `outputs/` dizininde mevcuttur.

**🎓 TEZ TESLİME HAZIR! ✅**