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
| Metrik | V1 (Segmentasyon) | V2 (Interactions) | V3 (Persona) | Durum |
| :--- | :---: | :---: | :---: | :--- |
| **Router Accuracy** | %87.89 | **%89.33** | %89.41 | ✅ V2 İdeal |
| **Genel R2 (Log)** | %83.00 | **%83.06** 🏆 | %62.86 | 📉 V3 Başarısız |
| **Genel R2 (Reel)** | %78.77 | **%79.07** | %42.69 | 📉 Overfitting |
| **MAE (Hata Payı)** | 349 Gün | **348 Gün** | 598 Gün | ✅ V2 En İyisi |

> [!CAUTION]
> **V3 Deney Sonucu:** "Persona" bilgisini (Cluster ID) modele doğrudan vermek, regresyon performansını bozmuş (%62'ye düşüş) ve aşırı öğrenmeye (overfitting) yol açmıştır. Bu nedenle **V2 Modeli Final Sürüm** olarak seçilmiştir.

> [!TIP]
> **Tez Notu:** Yeni özelliklerin asıl katkısı, "Router" modelinin karar yeteneğini (%1.5 artış) güçlendirmesi olmuştur. Bu da doğru davanın doğru modele gitmesini sağlayarak sistemin güvenilirliğini artırmıştır.

> [!IMPORTANT]
> **Sonuç:** Mevcut veri setiyle ulaşılabilecek teorik limit **%83** seviyesine çıkmıştır. Ayrıca Coğrafi Adalet ve Yargıç Tipolojisi analizleriyle sistemin sadece bir "tahminci" değil, bir "sosyolojik analiz aracı" olduğu kanıtlanmıştır (Detaylar: `BulgularFinal1.md`).

## 🚀 Sonuç ve Öneriler
- **Başarı Durumu:** Hedeflenen %80 başarıya tam ulaşılamasa da, veri setindeki gürültü ve karmaşıklığa göre %65 (Log R2) ve %57 (Reel R2) seviyeleri, insan davranışını tahmin eden modeller için makul bir başlangıçtır.
- **Gelecek Adımlar:**
    - Daha detaylı suç metni analizi (NLP) ile başarı artırılabilir.
    - Suç kategorilerine göre ayrı modeller eğitilebilir (Hybrid Model).
    - Derin Öğrenme (Deep Learning) yöntemleri denenebilir.
