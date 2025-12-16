# 🎓 Tez Dosya ve Görsel Rehberi

Bu rehber, tez yazım sürecinde hangi klasör ve dosyaların **GÜNCEL**, hangilerinin **ESKİ** olduğunu ayırt etmeniz için hazırlanmıştır.

---

## ✅ 1. ALTIN DEĞERİNDEKİ KLASÖRLER (Tez İçin Kullanılacaklar)

Bu klasörler `outputs/` dizini altındadır ve projenin en son, en bilimsel sonuçlarını içerir.

| Klasör Adı | Ne İçeriyor? | Tezde Nereye Konmalı? |
| :--- | :--- | :--- |
| 📂 **`scientific_analysis`** | - `theoretical_limit.png`: Modelin ulaşabileceği max başarı.<br>- `variance_decomposition.png`: Hatanın kaynağı (Veri mi Model mi?). | **Metodoloji / Tartışma** bölümüne. Modelin başarısının tesadüf olmadığını kanıtlar. |
| 📂 **`shap_analysis`** | - `shap_summary.png`: Modelin kararlarını etkileyen faktörler.<br>- `shap_race.png`: Irk değişkeninin etkisi. | **Açıklanabilirlik (Explainability)** bölümüne. Modelin "neden" karar verdiğini gösterir. |
| 📂 **`bias_analysis`** | - `race_bias.png`: Irklara göre hata oranları.<br>- `conditional_bias.png`: Suç şiddetine göre ırkçılık analizi. | **Etik ve Adalet (Bias & Fairness)** bölümüne. |
| 📂 **`clustering_analysis`** | - `cluster_pca_map.png`: Suçlu profilleri haritası.<br>- `cluster_profiles.csv`: "Genç-Tehlikeli" gibi grupların istatistiği. | **Bulgular / Keşifsel Analiz** bölümüne. Veri madenciliği yapıldığını gösterir. |
| 📂 **`geo_analysis`** | - `geo_justice_score.png`: Hangi ilçenin (County) daha sert olduğunu gösteren harita/grafik. | **Bulgular / Coğrafi Analiz** bölümüne. |
| 📂 **`judge_typology`** | - `judge_clusters.png`: Hakimlerin "Sert" ve "Yumuşak" olarak ayrışması. | **Bulgular / Sosyolojik Analiz** bölümüne. |
| 📂 **`high_severity_analysis`** | - `error_patterns.png`: Ağır suçlarda modelin nerede hata yaptığı. | **Model Performansı** bölümüne. |
| 📂 **`comprehensive_features`** | - Yeni üretilen kapsamlı özelliklerin listesi ve analizleri. | **Özellik Mühendisliği (Feature Engineering)** bölümüne. |

---

## ℹ️ 2. YARDIMCI KLASÖRLER (İsteğe Bağlı)

Bu klasörler de günceldir (Aralık 11-14) ancak tezde görsel olarak kullanılması şart değildir, ek bilgi verir.

*   📂 **`interaction_analysis`**: Özelliklerin birbiriyle etkileşimi (Teknik detay).
*   📂 **`router_classifier`**: Router modelinin (Sınıflandırıcı) iç detayları.
*   📂 **`explanation_analysis`**: CatBoost'un kendi feature importance grafikleri (SHAP varken buna gerek yok).

---

## ❌ 3. ESKİ KLASÖRLER (Kullanma / Arşiv)

Bu klasörler projenin ilk aşamalarından kalmadır. **Tezde kullanmanıza gerek yoktur**, kafa karışıklığı yaratabilir.

*   ❌ `00_yeni_baslangic`
*   ❌ `01_detayli_analiz`
*   ❌ `02_gelismis_analiz`
*   ❌ `model_results_v1`
*   ❌ `v3_persona_model` (Başarısız olan deneme)
*   ❌ `optimization_analysis`

---

## 📄 4. KÖK DİZİNDEKİ KRİTİK METİNLER

*   📜 **`BULGULAR_FINAL.md`**: Tezinizin "Bulgular" bölümünün taslağıdır. Buradaki metinleri kopyalayıp teze yapıştırabilirsiniz.
*   📜 **`WALKTHROUGH.md`**: Tezinizin "Yöntem" ve "Model Geliştirme" hikayesidir.
*   📜 **`succesful_new_copy/`**: Python kodlarını teze ekleyecekseniz SADECE bu klasördekileri kullanın.

---
*Hazırlayan: Antigravity AI Asistanı*
