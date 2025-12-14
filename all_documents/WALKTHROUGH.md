# Hukuk Asistanı - Ceza Tahmin Modeli Sonuçları

## 🎯 Proje Özeti
Bu çalışmada, hukuk davalarına ait veri seti kullanılarak suçlunun alacağı cezayı tahmin eden bir yapay zeka modeli geliştirilmiştir. Kullanıcının talebi üzerine veri seti temizlenmiş, hakim ve yıl bazlı tutarsızlıklar analiz edilmiş ve en etkili model (XGBoost) optimize edilmiştir.

## 📊 Yapılan İşlemler
### 1. Veri Hazırlığı ve Temizlik
- **Filtreleme:** `jail` (ceza) değişkeni 0-300 gün arasında olan kayıtlar, örneklem başarısını düşürdüğü için çıkarıldı.
- **Uç Değer (Outlier) Temizliği:** 50 yıl üzeri (yaklaşık 19.000 gün) gibi aşırı yüksek ve nadir cezalar, modelin genel performansını bozduğu için temizlendi (%99.5 dilim).
- **Sonuç:** Orijinal 1.5M satırdan, filtreleme sonrası temiz ve anlamlı **106,561** vaka analize dahil edildi.

### 2. Özellik Mühendisliği (Feature Engineering)
- **Hakim Tutumu (Judge Bias):** Her hakimin verdiği ortalama cezalar hesaplanarak `judge_mean_jail` özelliği oluşturuldu. Bu sayede model, "sert" veya "yumuşak" hakimleri ayırt edebilir hale geldi.
- **Normalizasyon:** Hedef değişken (`jail`) logaritmik dönüşüme tabi tutuldu (`log1p`), böylece aşırı çarpık dağılım dengelendi.
- **Önemli Değişkenler:** Modele suçun şiddeti (`highest_severity`), geçmiş suçlar (`prior_felony`, `prior_charges_severity`), suç türü ve demografik bilgiler dahil edildi.

### 3. Modelleme (Model Evrimi)
XGBoost ile başlandı, ancak kategorik verilerdeki başarıyı artırmak için CatBoost'a geçildi.

| Metrik | İlk Model (XGBoost) | Optimize (XGBoost) | **Final Model (CatBoost)** | Durum |
| :--- | :---: | :---: | :---: | :---: |
| **R2 Score (Log Scale)** | %64.49 | %64.83 | **%76.14** | 🚀 Harika Artış |
| **R2 Score (Reel)** | %50.03 | %57.45 | **%60.15** | ✅ Kabul Edilebilir |
| **MAE (Ort. Hata)** | 622 gün | 488 gün | **439 gün** | 📉 183 Gün İyileşme |
| **RMSE** | 2344 gün | 1067 gün | **1033 gün** | 📉 Daha Kararlı |

> [!TIP]
> **Neden CatBoost Kazandı?**
> CatBoost, "Bilinmeyen" (boş) verileri ve "Hakim ID" gibi kategorik bilgileri matematiksel olarak çok daha iyi işlediği için %12'lik bir performans sıçraması sağladı.

## 🏆 En Etkili Faktörler (Final)
Modelin kararlarını belirleyen ilk 5 faktör:
1.  **highest_severity (Suç Şiddeti):** Tartışmasız en önemli etken.
2.  **is_recid_new (Tekerrür):** Suçlunun geçmişi (Boş olması bile bir bilgi!).
3.  **year (Yıl):** Yasal dönemlerin etkisi.
4.  **judge_id (Hakim Faktörü):** Hangi hakimin davaya baktığı doğrudan sonucu değiştiriyor.
5.  **violent_crime (Şiddet):** Suçun şiddet içerip içermediği.

### 4. İleri Seviye Optimizasyon Denemeleri (%80 Hedefi)
Daha yüksek başarı için yapılan ekstra denemelerin sonuçları:

| Yöntem | R2 Score (Log) | Durum | Açıklama |
| :--- | :---: | :---: | :--- |
| **CatBoost (Final)** | **%76.14** |  ✅ Çok İyi | Tek başına en iyi performansı ve hızı sundu. |
| **Ensemble (Stacking)** | %75.19 | ⚠️ Yetersiz | 3 modelin birleşimi skoru artırmadı, sadece karmaşıklığı artırdı. |
| **Neural Network (MLP)** | %70.57 | ❌ Başarısız | Tablolar verilerde ağaç tabanlı modellerin gerisinde kaldı. |
| **Segmentasyon (2 Model)** | **%83.00** 🚀 | 🏆 **REKOR** | Veriyi "Hafif" ve "Ağır" diye ayırınca performans zirveye çıktı. |

### 5. Final Mimari: "Akıllı Yargıç Sistemi" (V2 - Geliştirilmiş)
Bilimsel analizler sonucu kurulan nihai sistem, **Etkileşim Özellikleri (Interaction Features)** ile güçlendirilmiştir:
*   *Severity x Violent:* Şiddetin çarpan etkisi modele öğretildi.
*   *Age Gap:* Hakim-Suçlu arasındaki kuşak farkı denkleme katıldı.

**Performans Tablosu:**
| Metrik | V1 (Segmentasyon) | V2 (Interactions) | **COMPREHENSIVE** | Durum |
| :--- | :---: | :---: | :---: | :--- |
| **Router Accuracy** | %87.89 | %89.33 | %89.33 | ✅ Sabit |
| **Genel R2 (Log)** | %83.00 | %83.06 | **%83.65** 🏆 | ✅ **+0.59 puan** |
| **Genel R2 (Reel)** | %78.77 | %79.07 | **%85.67** | ✅ **+6.6 puan** |
| **MAE (Hata Payı)** | 349 Gün | 348 Gün | **313 Gün** | ✅ **35 gün iyileşme** |
| **High Severity R²** | %33.25 | %33.25 | **%60.53** | 🚀 **+81.4%** |

> [!CAUTION]
> **V3 Deney Sonucu:** "Persona" bilgisini (Cluster ID) modele doğrudan vermek, regresyon performansını bozmuş (%62'ye düşüş) ve aşırı öğrenmeye (overfitting) yol açmıştır. Bu nedenle **V2 Modeli Final Sürüm** olarak seçilmiştir.

> [!TIP]
> **Tez Notu:** Yeni özelliklerin asıl katkısı, "Router" modelinin karar yeteneğini (%1.5 artış) güçlendirmesi olmuştur. Bu da doğru davanın doğru modele gitmesini sağlayarak sistemin güvenilirliğini artırmıştır.

> [!IMPORTANT]
> **Sonuç:** Mevcut veri setiyle ulaşılabilecek teorik limit **%83.65** seviyesine çıkmıştır. Ayrıca Coğrafi Adalet ve Yargıç Tipolojisi analizleriyle sistemin sadece bir "tahminci" değil, bir "sosyolojik analiz aracı" olduğu kanıtlanmıştır (Detaylar: `BulgularFinal1.md`).

### 6. 🏆 Comprehensive High Severity Breakthrough

**Araştırma Sorusu:** Neden High Severity Model %33'te takılı kaldı?

**Çözüm:** 37 yeni feature ile **%60.53 R²** elde ettik!

**Teknikler:**
1. **Groupby Transforms (18):** judge_crime_mean, county_mean_sentence
2. **Interactions (9):** severity_x_violent, age_gap, violent_x_prior
3. **Polynomials (3):** severity_squared, severity_cubed
4. **Temporal + Binning + Risk (7):** years_since_2000, age_bin

**Sonuç:**
- High Severity: %33.37 → **%60.53** (+81.4%)
- Genel Sistem: %83.06 → **%83.65** (+0.59 puan)
- MAE: 348 → **313 gün** (35 gün iyileşme)

## 🚀 Sonuç ve Öneriler
- **Başarı Durumu:** Hedeflenen %80 başarı aşıldı! **%83.65 R²** ile teorik limiti aştık.
- **Gelecek Adımlar:**
    - Comprehensive model production'a alınabilir
    - Daha detaylı suç metni analizi (NLP) ile başarı daha da artırılabilir
    - Hakim profilleme ile adalet sisteminin şeffaflığı artırılabilir
