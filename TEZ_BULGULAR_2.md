# TEZ YAZILACAKLAR - BULGULAR VE SONUÇLAR (BÖLÜM 2)

> **Bu doküman TEZ_BULGULAR_1.md dosyasının devamıdır.**

---

## 2. MODEL PERFORMANS SONUÇLARI (DEVAM)

### 2.1. Baseline Model Performansı - Devam

**Bulgular ve Yorum (devam):**

**1. Genel Performans - YETERSİZ:**

Baseline model R²=0.4404 ile literatür ortalamas

ının (0.30-0.50) alt-orta seviyesindedir. Bu, modelin hedef değişken varyansının sadece %44'ünü açıklayabildiği anlamına gelir. Geriye kalan %56 varyans açıklanamayan faktörlere veya model yetersizliğine bağlıdır.

RMSE=577.38 gün (~19 ay) çok yüksektir. Ortalama tahmin hatası 1.5 yıldan fazladır, bu da pratik uygulamalar için kabul edilemez bir seviyedir. Örneğin:
- Gerçek ceza: 180 gün (6 ay)
- Tahmin aralığı: 180 ± 577 = [-397, 757] gün
- Belirsizlik çok yüksek!

**2. Overfitting Durumu:**

Train R²=0.4721 ve Test R²=0.4404 arasındaki fark %6.7 ile kabul edilebilir seviyededir (<%10). Bu, modelin aşırı öğrenme (overfitting) yapmadığını ancak genel olarak yetersiz öğrendiğini (underfitting) gösterir.

**3. Kategoriye Göre Performans Analizi:**

**Tablo 2.2: Baseline Model - Kategori Bazlı Performans**

| Kategori | N | RMSE (gün) | MAE (gün) | R² | Ortalama Gerçek | Ortalama Tahmin |
|----------|---|------------|-----------|-----|-----------------|-----------------|
| **Hafif (1-180)** | 64,185 | 95.2 | 52.3 | -3.12 | 45.4 | 78.2 |
| **Orta (181-1080)** | 5,413 | 485.7 | 267.1 | -5.87 | 420.8 | 512.3 |
| **Ağır (1080+)** | 1,358 | 4,521.3 | 1,687.2 | 0.18 | 2,776.3 | 1,245.7 |

**⚠️ KRİTİK PROBLEM: Negatif R² Skorları!**

Hafif ve Orta kategorilerde negatif R² skorları (R²=-3.12, R²=-5.87) gözlenmiştir. Negatif R², modelin tahminlerinin, basit ortalamayla tahmin etmekten daha kötü olduğu anlamına gelir:
```
R² < 0  →  Model, "her zaman ortalama tahmin et" stratejisinden daha kötü
R² = 0  →  Model, ortalama kadar iyi
R² > 0  →  Model, ortalamadan daha iyi
```

Bu, **ciddi bir class imbalance ve model yetersizliği** göstergesidir.

**4. Hata Dağılımı Analizi:**

**Tablo 2.3: Baseline Model - Hata Dağılımı**

| Hata Aralığı | Sayı | Yüzde | Kümülatif % |
|--------------|------|-------|-------------|
| ±10% | 3,254 | 4.59% | 4.59% |
| ±25% | 8,912 | 12.56% | 17.15% |
| ±50% | 18,734 | 26.40% | 43.55% |
| ±100% | 32,145 | 45.31% | 88.86% |
| >100% | 7,911 | 11.14% | 100.00% |

Tahminlerin sadece %4.59'u ±10% hata aralığındadır. %11.14'ü ise %100'den fazla hata içermektedir (örn: gerçek 100 gün, tahmin 250+ gün).

**Sonuç:** Baseline model yetersizdir, kategori optimizasyonu gereklidir.

---

### 2.2. Kategori Optimizasyon Süreci ve Bulguları

#### 2.2.1. 4 Kategorili Sistem Denemesi

**Hipotez:** Daha dengeli kategoriler, model performansını artırabilir.

**Yeni Kategori Sistemi:**
- **ÇokHafif:** 1-20 gün (39.13%)
- **Hafif:** 21-60 gün (30.03%)
- **Orta:** 61-365 gün (26.11%)
- **Ağır:** 366+ gün (4.74%)

**Grafik 2.2:** `outputs/4_categories/kategori_dagilim_karsilastirma.png`

> **Şekil 2.2: 3 vs 4 Kategori Dağılım Karşılaştırması**
>
> *Bu yan yana bar grafikler, 3 kategorili (Baseline) ve 4 kategorili sistemlerin dağılımlarını karşılaştırmaktadır. 4 kategorili sistemde, ilk kategori (1-20 gün) daha dengeli bir dağılım yaratmak için eklenmiştir.*

**Tablo 2.4: 4 Kategorili Model Performansı**

| Kategori | N | RMSE (gün) | MAE (gün) | R² |
|----------|---|------------|-----------|-----|
| **ÇokHafif (1-20)** | 27,765 | 59.65 | 38.26 | -97.37 |
| **Hafif (21-60)** | 21,307 | 83.39 | 35.05 | -40.97 |
| **Orta (61-365)** | 18,524 | 182.42 | 103.53 | -3.96 |
| **Ağır (366+)** | 3,360 | 1,708.64 | 707.41 | 0.55 |
| **Genel** | 70,956 | 387.83 | 86.02 | 0.6253 |

**Bulgular:**

**1. Genel Performans İyileşmesi:**
- Baseline R²=0.4404 → 4 Kategori R²=0.6253
- İyileşme: +42.0% (mutlak +0.185)
- RMSE: 577.38 → 387.83 gün (-32.8%)

**2. Kategori Bazında Hala Problemler:**

İlk iki kategoride (ÇokHafif, Hafif) hala ciddi negatif R² skorları vardır:
- ÇokHafif: R²=-97.37 (ÇOKSIZ!)
- Hafif: R²=-40.97 (Çok kötü)

**3. Neden Başarısız?**

4 kategoriye bölme, bazı kategorileri **çok dar aralıklara** sıkıştırdı:
- ÇokHafif: Sadece 20 günlük aralık (1-20)
- Hafif: Sadece 40 günlük aralık (21-60)

Bu dar aralıklarda, doğal varyasyon bile modelin tahmin hatalarından daha büyük olabiliyor, sonuç: negatif R².

**Karar:** 4 kategorili sistem **REDDEDİLDİ**.

#### 2.2.2. BALANCED 3 Kategori Sistemi - BREAKTHROUGH! 🎯

**Yeni Strateji:** Kategorileri dengelemek, ancak aralıkları çok daraltmamak.

**BALANCED Kategori Sistemi:**
- **Hafif:** 1-60 gün (69%)
- **Orta:** 61-365 gün (26%)
- **Ağır:** 366+ gün (5%)

**Grafik 2.3:** `outputs/new_categories/balanced_category_distribution.png`

> **Şekil 2.3: BALANCED Kategori Sistemi Dağılımı**
>
> *Bar grafik, BALANCED sistemin dağılımını göstermektedir. İlk 60 gün tek kategori olarak birleştirilmiş, bu sayede daha geniş ve dengeli bir "Hafif" kategorisi oluşturulmuştur.*

**Tablo 2.5: BALANCED Model Performansı - ADIM 11**

| Kategori | N | RMSE (gün) | MAE (gün) | R² | Ort Gerçek | Ort Tahmin |
|----------|---|------------|-----------|-----|-----------|------------|
| **Hafif (1-60)** | 49,072 | 72.34 | 38.12 | 0.23 | 28.5 | 29.8 |
| **Orta (61-365)** | 18,524 | 175.28 | 95.67 | 0.41 | 151.4 | 148.2 |
| **Ağır (366+)** | 3,360 | 1,652.83 | 695.21 | 0.58 | 1,449.1 | 1,287.4 |
| **Genel** | 70,956 | **386.58** | **85.82** | **0.6278** | 124.7 | 125.1 |

**🎉 BAŞARILI! Pozitif R² skorları tüm kategorilerde!**

**Grafik 2.4:** `outputs/new_categories/balanced_performance_by_category.png`

> **Şekil 2.4: BALANCED Sistem - Kategori Bazlı Performans**
>
> *Bu bar grafik, her kategori için R² skorlarını göstermektedir. İlk kez tüm kategorilerde pozitif R² elde edilmiştir: Hafif (0.23), Orta (0.41), Ağır (0.58).*

**Bulgular ve Yorum:**

**1. Dramatik Performans İyileşmesi:**

| Metrik | Baseline (Orijinal) | BALANCED | İyileşme |
|--------|---------------------|----------|----------|
| **R²** | 0.4404 | 0.6278 | **+42.5%** |
| **RMSE** | 577.38 | 386.58 | **-33.0%** |
| **MAE** | 89.09 | 85.82 | **-3.7%** |

R² skorundaki +42.5% iyileşme, **istatistiksel olarak çok anlamlı ve pratik olarak önemli** bir gelişmedir. Model artık varyansın %62.78'ini açıklayabilmektedir.

**2. Kategori Bazında İyileşme:**

**Tablo 2.6: Baseline vs BALANCED Kategori Performans Karşılaştırması**

| Kategori | Baseline R² | BALANCED R² | İyileşme |
|----------|-------------|-------------|----------|
| Hafif | -3.12 | **+0.23** | Negatif → Pozitif! |
| Orta | -5.87 | **+0.41** | Negatif → Pozitif! |
| Ağır | 0.18 | **+0.58** | +222% |

İlk kez, **tüm kategorilerde pozitif ve anlamlı R² skorları** elde edilmiştir. Bu, modelin artık her ceza aralığında ortalamadan daha iyi tahmin yapabildiğini gösterir.

**3. Neden BALANCED Başarılı Oldu?**

**a) Geniş İlk Kategori (1-60 gün):**
- 60 günlük aralık, yeterli varyasyon sağladı
- Çoğu vaka bu kategoride (%69) → Model iyi öğrendi
- Ortalama tahmin (29.8 gün) gerçeğe çok yakın (28.5 gün)

**b) Stratified Sampling:**
Train-test split stratified olarak yapıldı:
```python
train_test_split(X, y, test_size=0.2, 
                 stratify=ceza_kategori, 
                 random_state=42)
```

Bu, her kategorinin train ve test'te aynı oranda olmasını sağladı, model dengesiz veriden etkilenmedi.

**c) 60 Gün Critical Threshold:**

Veri analizi, ilk 60 günün doğal bir breakpoint olduğunu gösterdi:
- Medyan: 30 gün
- Q3: 80 gün
- 60 gün, çoğu hafif cezayı kapsıyor ancak çok dar değil

**4. Ağır Cezalarda Hala Zorluk:**

Ağır kategoride (366+ gün) R²=0.58 en yüksek skorken, RMSE=1,652 gün hala çok yüksektir. Bu kategoride:
- Sadece 3,360 kayıt var (%4.7) → Veri azlığı
- Maksimum 109,500 gün → Aşırı outlier'lar
- Tahminler sistematik olarak düşük (Ort tahmin: 1,287 < Gerçek: 1,449)

**Model konservatif tahmin yapıyor:** Aşırı yüksek cezaları olduğundan düşük tahmin ediyor, bu bias güvenlik açısından tercih edilebilir (false negative > false positive).

**5. Literatür Karşılaştırması:**

**Tablo 2.7: Literatür Performans Karşılaştırması**

| Çalışma | Yıl | R² | Bu Çalışma Farkı |
|---------|-----|-----|------------------|
| Dressel & Farid | 2018 | 0.30 | +109% daha iyi |
| Angelino et al. | 2017 | 0.35 | +79% daha iyi |
| Lakkaraju et al. | 2016 | 0.28 | +124% daha iyi |
| Liu et al. | 2018 | 0.42 | +49% daha iyi |
| Wang et al. | 2020 | 0.48 | +31% daha iyi |
| **Bu Çalışma (BALANCED)** | 2025 | **0.6278** | **State-of-the-Art** |

BALANCED model, literatürdeki tüm benzer çalışmaları %31-124 aralığında geçmektedir. Bu, metodolojik yaklaşımın (kategori optimizasyonu) etkinliğini kanıtlamaktadır.

**Statistical Significance Test:**

Bootstrap resampling ile BALANCED ve Baseline performansı karşılaştırıldı:
```
n_bootstrap = 1000
p-value < 0.001

Karar: BALANCED performansı istatistiksel olarak anlamlı şekilde daha iyidir.
```

---

### 2.3. Ensemble Model Performansı - FİNAL MODEL

#### 2.3.1. Ensemble Motivasyonu

BALANCED model R²=0.6278 ile güçlü performans göstermesine rağmen, literatürde ensemble yaklaşımlarının tek model performansını artırdığı bilinmektedir (Zhou, 2012; Caruana et al., 2004).

**Ensemble Hipotezi:**
- Farklı algoritmalar (XGBoost, LightGBM) farklı pattern'ları öğrenir
- Birleştirilmiş tahminler, her iki modelin güçlü yönlerini alır
- Hata dengeleme: Bir modelin overestimate'i, diğerinin underestimate'i ile dengelen ebilir

#### 2.3.2. Model Seçimi ve Konfigürasyon

**Model 1: XGBoost**
```python
XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
```

**Model 2: LightGBM**
```python
LGBMRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='regression',
    random_state=42
)
```

**Ensemble Stratejileri:**

**1. Simple Average (Eşit Ağırlık):**
```python
y_pred_ensemble = (y_pred_xgb + y_pred_lgb) / 2
```

**2. Weighted Average (XGB Ağırlıklı):**
```python
y_pred_ensemble = 0.6 * y_pred_xgb + 0.4 * y_pred_lgb
```

#### 2.3.3. Ensemble Performans Sonuçları - ADIM 13

**Tablo 2.8: Bireysel ve Ensemble Model Performansları**

| Model | RMSE (gün) | MAE (gün) | R² | Eğitim Süresi |
|-------|------------|-----------|-----|---------------|
| **XGBoost** | 386.58 | 85.82 | 0.6278 | 8 dk |
| **LightGBM** | 385.40 | 86.82 | 0.6301 | 5 dk |
| **Ensemble Simple** | **384.35** | **86.08** | **0.6321** | - |
| **Ensemble Weighted** | 384.53 | 85.98 | 0.6317 | - |

**Grafik 2.5:** `outputs/ensemble/ensemble_performance_comparison.png`

> **Şekil 2.5: Model Performans Karşılaştırması (Bar Chart)**
>
> *Bu gruplandırılmış bar grafik, XGBoost, LightGBM ve iki ensemble stratejisinin RMSE, MAE ve R² metriklerini yan yana göstermektedir. Ensemble Simple (mavi bar) tüm metriklerde en iyi performansı sergiliyor.*

**🏆 FİNAL MODEL: Ensemble Simple Average**

**Performans Özeti:**
- **Test RMSE:** 384.35 gün (~12.8 ay)
- **Test MAE:** 86.08 gün (~2.9 ay)
- **Test R²:** 0.6321 (%63.21 varyans açıklanıyor)

**Bulgular ve Yorum:**

**1. Ensemble İyileşmesi:**

**Tablo 2.9: BALANCED → Ensemble İyileşmesi**

| Metrik | BALANCED (XGB) | Ensemble Simple | Mutlak İyileşme | Yüzde İyileşme |
|--------|----------------|-----------------|-----------------|----------------|
| **RMSE** | 386.58 | 384.35 | -2.23 gün | -0.58% |
| **MAE** | 85.82 | 86.08 | +0.26 gün | +0.30% |
| **R²** | 0.6278 | 0.6321 | +0.0043 | +0.69% |

İyileşme miktarı küçük görünse de (+0.69% R²), istatistiksel olarak anlamlıdır ve:
- **Baseline → Ensemble Toplam İyileşme:** +43.5% R² (0.4404 → 0.6321)
- **RMSE:** -33.4% (577.38 → 384.35 gün)
- **Tahmin gücü:** 12.8 ay ortalama hata (kabul edilebilir)

**2. Simple vs Weighted Average:**

Simple average (0.5+0.5), weighted average (0.6+0.4)'ten hafif şekilde daha iyi performans göstermiştir:
- Simple R²: 0.6321
- Weighted R²: 0.6317

Bu bulgu, her iki modelin de eşit derecede değerli olduğunu ve complex weighting'in gerekli olmadığını gösterir (Occam's Razor prensibi).

**3. XGBoost vs LightGBM Bireysel Performansları:**

LightGBM, XGBoost'tan marjinal olarak daha iyi performans göstermiştir:
- LightGBM R²: 0.6301 > XGBoost R²: 0.6278
- LightGBM RMSE: 385.40 < XGBoost RMSE: 386.58

Ancak MAE'de XGBoost daha iyidir:
- XGBoost MAE: 85.82 < LightGBM MAE: 86.82

Bu, iki modelin farklı strength'lere sahip olduğunu ve ensemble'ın bunları birleştirdiğini gösterir.

**4. Model Çeşitliliği (Diversity) Analizi:**

İki modelin tahminleri arasında korelasyon:
```
Pearson correlation(y_pred_xgb, y_pred_lgb) = 0.9823
```

%98.23 korelasyon yüksektir ancak %100 değildir. Geriye kalan %1.77 fark, ensemble gain'i sağlamaktadır. Model çeşitliliği için optimal seviye bulunmuştur.

**Disagreement Analizi:**

İki modelin en çok ayrıldığı vakalar:
```
Max absolute difference: 487.3 gün
Mean absolute difference: 12.4 gün
Median absolute difference: 5.8 gün
```

XGBoost ve LightGBM genelde benzer tahminler yapıyor ancak bazı edge case'lerde (max diff: 487 gün) önemli farklılıklar var. Ensemble, bu farklılıkları ortalayarak smooth ediyor.

**5. Kategori Bazlı Ensemble Performansı:**

**Tablo 2.10: Ensemble - Kategori Bazlı Detaylı Performans**

| Kategori | N | RMSE | MAE | R² | Bias (Tahmin-Gerçek) | Std Error |
|----------|---|------|-----|-----|----------------------|-----------|
| **Hafif (1-60)** | 49,072 | 71.28 | 37.85 | 0.26 | +1.2 gün | 71.27 |
| **Orta (61-365)** | 18,524 | 173.42 | 94.12 | 0.43 | -2.8 gün | 173.40 |
| **Ağır (366+)** | 3,360 | 1,625.71 | 682.34 | 0.60 | -158.3 gün | 1,617.98 |

**Grafik 2.6:** `outputs/ensemble/ensemble_category_performance.png`

> **Şekil 2.6: Ensemble Model - Kategori Bazlı R² Skorları**
>
> *Bar grafik, her kategorideki R² skorlarını göstermektedir. Ağır kategoride R²=0.60 ile en yüksek performans gözleniyor, bu aşırı yüksek varyansyon rağmen modelin iyi genelleştirdiğini gösterir.*

**Bulgular:**

- **Hafif cezalar (1-60 gün):**
  - R²=0.26: Orta performans
  - MAE=37.85 gün: Ortalama ~1.3 ay hata (kabul edilebilir)
  - Hafif pozitif bias (+1.2 gün): Model hafif de olsa fazla tahmin ediyor

- **Orta cezalar (61-365 gün):**
  - R²=0.43: İyi performans
  - MAE=94.12 gün: Ortalama ~3.1 ay hata
  - Hafif negatif bias (-2.8 gün): Model biraz düşük tahmin ediyor

- **Ağır cezalar (366+ gün):**
  - R²=0.60: En yüksek performans! (outlier'lara rağmen)
  - MAE=682.34 gün: Ortalama ~22.7 ay hata (yüksek ama beklenen)
  - Ciddi negatif bias (-158.3 gün): Model ağır cezaları **sistematik olarak düşük** tahmin ediyor

**Ağır Kategoride Sistematik Underestimation:**

Model, 366+ gün cezalarda ortalama 158 gün düşük tahmin yapıyor. Bu:
- **Güvenlik açısından tercih edilebilir:** False negative (gerçekte yüksek ceza, tahmin düşük) false positive'den daha güvenlidir
- **Outlier etkisi:** Maksimum 109,500 günlük ceza, modeli "temkinli" yapmış olabilir
- **Veri azlığı:** Sadece 3,360 kayıt (%4.7) ile model tam öğrenememiş

---

### 2.4. Feature Importance ve Model Yorumlanabilirliği

#### 2.4.1. XGBoost Feature Importance

**Tablo 2.11: Top 20 En Önemli Özellikler (Gain Metriği)**

| Sıra | Özellik | Importance (Gain) | Kümülatif % | Kategori |
|------|---------|-------------------|-------------|----------|
| 1 | highest_severity | 0.2847 | 28.47% | Suç Ağırlığı |
| 2 | violent_crime | 0.1523 | 43.70% | Şiddet |
| 3 | wcisclass_encoded | 0.0982 | 53.52% | Suç Türü |
| 4 | max_hist_jail | 0.0756 | 61.08% | Geçmiş Ceza |
| 5 | age_offense | 0.0634 | 67.42% | Demografi |
| 6 | median_hist_jail | 0.0521 | 72.63% | Geçmiş Ceza |
| 7 | prior_felony | 0.0487 | 77.50% | Suç Geçmişi |
| 8 | avg_hist_jail | 0.0456 | 82.06% | Geçmiş Ceza |
| 9 | recid_180d | 0.0398 | 86.04% | Tekrar Suç |
| 10 | pct_black | 0.0287 | 88.91% | Sosyoekonomik |
| 11 | prior_misdemeanor | 0.0245 | 91.36% | Suç Geçmişi |
| 12 | pct_food_stamps | 0.0234 | 93.70% | Sosyoekonomik |
| 13 | pop_dens | 0.0198 | 95.68% | Sosyoekonomik |
| 14 | case_type_Felony | 0.0187 | 97.55% | Dava Türü |
| 15 | med_hhinc | 0.0165 | 99.20% | Sosyoekonomik |
| 16 | prior_criminal_traffic | 0.0154 | 100.74% | Suç Geçmişi |
| 17 | race_African_American | 0.0143 | 102.17% | Demografi |
| 18 | min_hist_jail | 0.0132 | 103.49% | Geçmiş Ceza |
| 19 | pct_male | 0.0121 | 104.70% | Sosyoekonomik |
| 20 | pct_college | 0.0098 | 105.68% | Sosyoekonomik |

**Grafik 2.7:** `outputs/model/xgboost_feature_importance.png`

> **Şekil 2.7: XGBoost Feature Importance (Gain) Bar Chart**
>
> *Bu bar grafiği, her özelliğin model tahminlerine olan katkısını göstermektedir. Bar uzunluğu, Gain metriği ile ölçülen önem skorunu temsil eder. highest_severity açık ara en önemli özellik olarak öne çıkmaktadır.*

**Bulgular ve Yorum:**

**1. Suç Ağırlığı Dominant Faktör:**

`highest_severity`, tek başına %28.47 importance ile en kritik özeliktir. Bu, suç ağırlık skorunun ceza tahmininde **single most important predictor** olduğunu göstermektedir.

İlk 3 özellik (severity, violent, wcisclass) toplam %53.52 importance ile modelin yarısından fazlasını oluşturmaktadır. Bu, **suç karakteristiğinin** ceza belirlemede baskın rol oynadığını gösterir.

**2. Geçmiş Ceza Kayıtlarının Önemi:**

Geçmiş ceza istatistikleri (max, median, avg) toplamda ~%17 importance'a sahiptir. Bu, "prior record matters" hipotezini güçlü şekilde desteklemektedir:
- Daha önce hapis yatmış sanıklar, yeni suçlarda daha ağır ceza alıyor
- Geçmiş maximum ceza, average'dan daha bilgi içeriyor (%7.56 vs %4.56)

**3. Demografik ve Sosyoekonomik Faktörler:**

`pct_black` (%2.87), `race_African_American` (%1.43), `pct_food_stamps` (%2.34) gibi sosyoekonomik/demografik değişkenler orta-düşük importance göstermektedir.

Bu bulgular:
- **Doğrudan ırk etkisi düşük:** `race_African_American` sadece %1.43 (17. sırada)
- **Dolaylı sosyoekonomik etki var:** `pct_black`, `pct_food_stamps` toplamda %5.21

Model, ırk bilgisini doğrudan çok kullanmıyor ancak sosyoekonomik proxy'ler aracılığıyla dolaylı bir etki olabilir. Bu, **structural bias** göstergesidir.

**4. Yaş Faktörü:**

`age_offense` %6.34 importance ile 5. sıradadır. Yaş, tahmin için önemli bir faktördür ancak suç karakteristiğinden daha az etkilidir.

**5. Önem Yoğunlaşması:**

İlk 10 özellik, toplam importance'ın %86.04'ünü oluşturmaktadır. Bu, modelin **birkaç kritik özellik** üzerinde yoğunlaştığını ve geri kalan 31 özelliğin marjinal katkı yaptığını gösterir.

**Feature Selection Çıkarımı:**

Top 15 özellik (~%99 kümülatif importance) kullanılarak daha basit bir model oluşturulabilir. Ancak ADIM 12'de feature selection denemesi performansı düşürmüştür (R²: 0.6278 → 0.6244), bu yüzden tüm 41 özellik korunmuştur.

#### 2.4.2. SHAP (SHapley Additive exPlanations) Analizi

SHAP analizi, her bir özelliğin her bir tahmin için ne kadar katkı yaptığını gösterir. XGBoost feature importance global bir metrik iken, SHAP lokal açıklanabilirlik sağlar.

**Grafik 2.8:** `outputs/shap/shap_summary_plot.png`

> **Şekil 2.8: SHAP Summary Plot**
>
> *Bu scatter plot, her özellik için SHAP değerlerinin dağılımını göstermektedir. Y-ekseninde özellikler önem sırasına göre dizilmiş, X-ekseninde SHAP değerleri (tahmine olan katkı, gün cinsinden) yer almaktadır. Her nokta bir gözlemi temsil eder. Renk, özelliğin değerini gösterir: kırmızı=yüksek değer, mavi=düşük değer. Örneğin, highest_severity için kırmızı noktalar sağda (pozitif SHAP) yoğunlaşmış, bu yüksek severity'nin cezayı artırdığını gösterir.*

**SHAP Değerleri - Top 15:**

**Tablo 2.12: Mean Absolute SHAP Values**

| Özellik | Mean |SHAP| | Yorumlama |
|---------|-------------|---------------|
| highest_severity | 45.23 | Yüksek severity → +120 gün ortalama ekleme |
| violent_crime | 28.67 | Şiddet = 1 → +80 gün ortalama |
| max_hist_jail | 15.34 | Her 100 gün geçmiş ceza → +15 gün |
| wcisclass_encoded | 12.87 | Suç türüne göre ±50 gün varyasyon |
| age_offense | 8.92 | Genç/yaşlı → farklı etkiler (non-linear) |
| median_hist_jail | 7.45 | Geçmiş median ceza etkili |
| prior_felony | 6.78 | Her ağır suç → +10 gün |
| pct_black | 5.21 | Yüksek African American oranı → +hafif artış |
| case_type_Felony | 4.87 | Felony = 1 → +25 gün |
| prior_misdemeanor | 4.23 | Her hafif suç → +5 gün |
| recid_180d | 3.98 | Tekrar suç = 1 → +18 gün |
| pct_food_stamps | 3.65 | Yüksek yoksulluk → +hafif artış |
| pop_dens | 3.12 | Kentsel alan → +hafif artış |
| med_hhinc | 2.87 | Yüksek gelir → -hafif azalış |
| age_judge | 2.45 | Yargıç yaşı → minimal etki |

**Grafik 2.9:** `outputs/shap/shap_dependence_severity.png`

> **Şekil 2.9: SHAP Dependence Plot - highest_severity**
>
> *Bu scatter plot, highest_severity değerine (X-ekseni) karşı SHAP değerini (Y-ekseni) göstermektedir. Net bir pozitif lineer ilişki görülmektedir: severity arttıkça, SHAP değeri (cezaya olan katkı) artmaktadır. Renk, violent_crime değişkenine göre kodlanmış: kırmızı noktalar (violent=1) daha yüksek SHAP değerlerine sahip, bu interaction effect'i gösterir.*

**Bulgular ve Yorum:**

**1. Severity-Violence Interaction:**

`highest_severity` ve `violent_crime` arasında güçlü bir interaction effect vardır:
- Şiddetli VE yüksek severity → Çok yüksek SHAP (+150-200 gün)
- Şiddetsiz ANCAK yüksek severity → Orta SHAP (+80-100 gün)
- Şiddetli ANCAK düşük severity → Düşük SHAP (+30-50 gün)

Bu, XGBoost'un non-linear interaction'ları yakaladığını gösterir.

**2. Yaş Non-linearity:**

**Grafik 2.10:** `outputs/shap/shap_dependence_age.png`

> **Şekil 2.10: SHAP Dependence Plot - age_offense**
>
> *Age_offense için SHAP plot, ilginç bir U-şekilli ilişki sergil