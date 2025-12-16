# 🎓 TEZ ARAŞTIRMA BULGULARI VE SONUÇ RAPORU
**Tarih:** 15.12.2025

Bu rapor, tez yazım şablonunun 4. ve 5. bölümlerinde kullanılacak **KESİN** verileri içerir.

---

## 4. ARAŞTIRMA BULGULARI

### 4.1. Model Performans Özeti (Final Sistem)
| Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **R² (Log Scale)** | **%83.65** | Modelin genel açıklayıcılık gücü (Çok Yüksek) |
| **R² (Reel Scale)** | **%85.67** | Gerçek gün bazında tahmin başarısı |
| **MAE (Ortalama Hata)** | **313 Gün** | Ortalama sapma miktarı |
| **Router Başarısı** | **%89.33** | Davaları doğru modele yönlendirme oranı |

### 4.2. Kritik İyileştirme (High Severity Breakthrough)
Tezin en güçlü yanı, ağır suçları tahmin etme başarısındaki artıştır:
*   **Eski Başarı:** %33.37
*   **Yeni Başarı:** **%60.53**
*   **Artış:** +%81.4 İyileşme 🚀

### 4.3. En Önemli Faktörler (SHAP Analizi)
Modelin kararlarını etkileyen ilk 5 faktör:
1.  **violent_recid (YENİ):** Şiddet suçu ve sabıka birlikteliği.
2.  **highest_severity:** Suçun yasal tanımındaki ağırlık.
3.  **is_recid_new:** Tekerrür durumu.
4.  **wcisclass:** Suç sınıfı kodu.
5.  **severity_x_violent:** Şiddet ve ciddiyet etkileşimi.

---

## 5. SONUÇ VE ÖNERİLER İÇİN VERİLER

### 5.1. Etik Analiz (Bias) Sonuçları
*   **Irk Yanlılığı:** Model, Siyahilere (African American) ortalama **57 gün**, Beyazlara (Caucasian) **48 gün** EKSİK ceza tahmin etmektedir. Sistematik bir ırkçılık (bir gruba aşırı ceza verme) gözlemlenmemiştir.
*   **Cinsiyet Yanlılığı:** Erkekler, kadınlara göre daha yüksek ceza tahminleri almaktadır.

### 5.2. Coğrafi Adalet
*   Bazı ilçeler (County 54) diğerlerine göre sistematik olarak daha sert kararlar vermektedir (+193 Gün).

---

## 📂 KLASÖR REHBERİ (Hangi Dosya Nereye?)

*   **4.1. Veri Analizi:** `4.1_Veri_Analizi` klasöründeki `eda_*.png` grafikleri.
*   **4.2. Model Yapısı:** `4.2_Model_Mimarisi` klasöründeki `router_classifier_*.png` görselleri.
*   **4.3. Bulgular:** `4.3_Performans_Bulgulari` içindeki `scientific_analysis_*.png` grafikleri.
*   **4.4. Tartışma:** `4.4_Aciklanabilirlik` içindeki `shap_analysis_summary_plot.png`.
*   **4.5. Etik:** `4.5_Etik_ve_Adalet` içindeki `bias_analysis_race_bias.png`.

