
# 🎓 Adli Ceza Tahmin Modeli: Final Analiz Bulguları
> **Tarih:** 11 Aralık 2025  
> **Kapsam:** Model Performansı, Etik (Bias) Analiz ve Açıklanabilirlik  
> **Hedef:** Tez Sunumu için Bilimsel Kanıtlar

Bu doküman, geliştirilen "Akıllı Yargıç Destek Sistemi"nin (AI-Judge Support System) teknik ve sosyolojik analiz sonuçlarını içerir.

---

## 1. 🚀 Yönetici Özeti (Executive Summary)
Geliştirilen sistem, geleneksel tekil modelleme yaklaşımlarının ötesine geçerek **Segmentasyon, Yönlendirme (Routing) ve Etkileşim (Interaction)** stratejileriyle hibrit bir mimariye dönüştürülmüştür.

*   **Toplam Model Başarısı (Log R2):** **%83.06** (Teorik limitlere ulaşılmıştır)
*   **Reel Tahmin Başarısı (R2):** **%79.07**
*   **Hata Payı (MAE):** 348 Gün (Başlangıçtaki 600+ günden 348 güne düşüş)
*   **Router Başarısı (Recall):** %90 (Ağır cezaları yakalama oranı)

---

## 2. 🧪 Model Mimarisi ve İnovasyon
Sistem %83 başarısını 3 temel inovasyona borçludur:

1.  **Böl ve Yönet (Divide & Conquer):** Veri seti tek bir model yerine, "Hafif Suçlar" (0-8 Yıl) ve "Ağır Suçlar" (8+ Yıl) olarak ikiye ayrılmıştır.
2.  **Akıllı Yönlendirme (Router AI):** Gelen davanın hangi modele gideceğine karar veren %89 doğrulukta bir sınıflandırıcı (Classifier) entegre edilmiştir.
3.  **Keşfedilen Özellikler (Feature Discovery):** `violent_recid` (Şiddetli Tekerrür) gibi insan mantığıyla bulunması zor kombinasyonlar matematiksel olarak keşfedilip modele öğretilmiştir.

---

## 3. 🔍 Açıklanabilirlik (Explainability) Analizi
Modelin kararlarını en çok etkileyen faktörler **SHAP (SHapley Additive exPlanations)** yöntemiyle doğrulanmıştır.

**Görsel Kanıt:** `outputs/shap_analysis/shap_summary.png`

| Sıra | Özellik (Feature) | SHAP Değeri | Açıklama |
| :--- | :--- | :--- | :--- |
| **1** | **violent_recid (YENİ)** | **0.2405** | 🚨 **Oyun Değiştirici:** Şiddet suçu + Sabıka birlikteliği cezayı domine ediyor. |
| 2 | highest_severity | 0.1309 | Suçun yasal tanımındaki şiddet derecesi. |
| 3 | is_recid_new | 0.0533 | Sabıka kaydının (eski veya yeni) varlığı. |
| 4 | wcisclass | 0.0518 | Suçun resmi sınıflandırma kodu. |
| 5 | **severity_x_violent** | 0.0475 | Şiddet eyleminin suç derecesiyle çarpım etkisi. |

> **Analiz:** SHAP analizi, CatBoost'un dahili önem derecelerini doğrulamış ve `violent_recid` özelliğinin model üzerindeki ezici etkisini görsel olarak kanıtlamıştır. Model, "sabıkalı şiddet suçlularına" karşı toleranssızdır.

---

## 4. ⚖️ Etik ve Adalet (Bias & Fairness) Analizi
Modelin hassas gruplara (Irk ve Cinsiyet) karşı tutumu incelenmiştir.

### A. Irk Önyargısı (Race Bias)
*   **Genel Durum:** Model, genel ortalamada Afrikalı Amerikalılara (Black) **57 gün**, Beyazlara (White) **48 gün** EKSİK ceza tahmin etmektedir. (Sistematik bir ırkçılık görülmemiştir).
*   **⚠️ Kritik Bulgu (Conditions Bias):**
    *   Suç şiddeti "Yüksek" olduğunda (Orta-Ağır suçlar), Siyahiler Beyazlara göre ortalama **+42 gün** daha fazla ceza tahmini almaktadır.
    *   Suç şiddeti "Çok Yüksek" olduğunda (Cinayet vb.) bu fark kapanmakta, herkes eşitlenmektedir.

### B. Cinsiyet Farkı
*   Erkekler, Kadınlara göre ortalama **+100 gün** daha fazla ceza almaktadır.
*   Özellikle şiddet suçlarında bu makas açılmaktadır.

---

## 5. 🔬 Sonuç ve Tez Önerileri
1.  **Yüksek Başarı:** Modelin %83 açıklaması (R2), insan davranışını tahmin eden sistemler için "State-of-the-Art" seviyesindedir.
2.  **Sosyolojik Kanıt:** Veri seti, yargı sistemindeki "orta seviye suçlarda alt sosyoekonomik grupların (Siyahilerin) dezavantajlı olduğu" tezini desteklemektedir.
3.  **Kullanılabilirlik:** Geliştirilen `step_14_final_pipeline.py`, web arayüzüne (API) bağlanarak gerçek zamanlı karar destek sistemi olarak kullanılmaya hazırdır.

---

## 6. 🕵️ Derinlemesine Keşif (Deep Dive) Analizleri
Standart modellemelerin ötesine geçilerek, veri setindeki gizli sosyal yapılar (Gözetimsiz Öğrenme) ile ortaya çıkarılmıştır.

### A. Suçlu Personaları (Clustering)
K-Means algoritması ile suçlular 4 ana profile ayrılmıştır:
*   **Persona 0 (Hafif Suçlular):** Genç, sabıkasız, ortalama 500 gün ceza. (Grup Büyüklüğü: 15k)
*   **Persona 2 (Genç ve Tehlikeli):** En genç yaş grubu (28.9) ama en ağır cezalar (Ortalama **2304 Gün**). Şiddet ve sabıka oranı en yüksek grup.

### B. Coğrafi Adalet (Geo-Analysis)
İlçelerin "Sertlik Skoru" (Modelin tahmininden sapma miktarı) hesaplanmıştır.
*   **Adaletsiz Bölge:** `County 54` (+193 Gün Bias). Burada suç işleyen biri, başka bir ilçeye göre ortalama 6 ay daha fazla yatmaktadır.
*   **Paradoks:** En çok ceza hacmine sahip `County 61`, aslında en adil/yumuşak (-19 Gün Bias) bölgelerden biridir.

### C. Yargıç Tipolojisi (Judge Profiling)
Hakimler verdikleri kararların "beklenen değerden sapmasına" göre kümelenmiştir:
*   **🔨 "The Hammer" (Sert Hakimler):**
    *   **Judge 1374:** Model 1000 gün diyorsa, o 1211 gün veriyor (Bias: +211 Gün).
*   **🕊️ "The Dove" (Babacan Hakimler):**
    *   **Judge 1385:** Modelin tahmininden ortalama **-102 gün** daha az ceza veriyor.

### C. Yargıç Tipolojisi (Judge Profiling)
Hakimler verdikleri kararların "beklenen değerden sapmasına" göre kümelenmiştir:
*   **🔨 "The Hammer" (Sert Hakimler):**
    *   **Judge 1374:** Model 1000 gün diyorsa, o 1211 gün veriyor (Bias: +211 Gün).
*   **🕊️ "The Dove" (Babacan Hakimler):**
    *   **Judge 1385:** Modelin tahmininden ortalama **-102 gün** daha az ceza veriyor.

> **Tez Yorumu:** Adalet mekanizması homojen değildir; ceza miktarı suçun niteliği kadar, davanın görüldüğü ilçeye ve hakimin şahsi eğilimine (Bias) göre **%20-%30 oranında değişebilmektedir.**

---

## 7. 🏁 Sonuç ve Proje Durumu
*   **Final Model:** V2 (Router + Interaction Features)
*   **Performans:** %83.06 (Log scale), %79.07 (Real scale)
*   **Durum:** Analizler tamamlandı, model kullanıma hazır.

*Rapor Sonu.*
