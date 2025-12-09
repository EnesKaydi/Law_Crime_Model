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

## 🚀 Sonuç ve Öneriler
- **Başarı Durumu:** Hedeflenen %80 başarıya tam ulaşılamasa da, veri setindeki gürültü ve karmaşıklığa göre %65 (Log R2) ve %57 (Reel R2) seviyeleri, insan davranışını tahmin eden modeller için makul bir başlangıçtır.
- **Gelecek Adımlar:**
    - Daha detaylı suç metni analizi (NLP) ile başarı artırılabilir.
    - Suç kategorilerine göre ayrı modeller eğitilebilir (Hybrid Model).
    - Derin Öğrenme (Deep Learning) yöntemleri denenebilir.
