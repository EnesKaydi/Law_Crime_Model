# İÇİNDEKİLER

## 3. MATERYAL VE YÖNTEMLER

### 3.1. Materyal (Veri Seti)
   3.1.1. Veri Kaynağı ve İçeriği
   3.1.2. Veri Yapısı ve Kolonların Açıklamaları
   3.1.3. Veri Temizliği ve Seçim Süreci
   3.1.4. Keşifsel Veri Analizi (EDA) Yöntemi

### 3.2. Yöntemler
   3.2.1. Özellik Mühendisliği (Feature Engineering)
   3.2.2. Etiket (Label) Belirleme
   3.2.3. Model Geliştirme Süreci
      3.2.3.1. Baseline Model
      3.2.3.2. Kategori Optimizasyonu
      3.2.3.3. Hyperparameter Tuning
      3.2.3.4. Ensemble Model Tasarımı
   3.2.4. XGBoost Algoritması ve Tercih Gerekçeleri
   3.2.5. Model Değerlendirme Metrikleri
   3.2.6. Açıklanabilirlik Analizi (SHAP)
   3.2.7. Ceza-İş Atama Sistemi
   3.2.8. Fairness ve Bias Değerlendirmesi
   3.2.9. Sistem Altyapısı ve Teknolojik Mimari

---

## 4. ARAŞTIRMA BULGULARI VE TARTIŞMA

### 4.1. Keşifsel Veri Analizi Bulguları
   4.1.1. Veri Seti Genel Karakteristikleri
   4.1.2. Hedef Değişken (Jail) Dağılımı ve İstatistikleri
   4.1.3. Demografik Değişkenlerin Analizi
      4.1.3.1. Cinsiyet Dağılımı
      4.1.3.2. Irk/Etnik Köken ve Sistemik Bias
      4.1.3.3. Dava Türü Dağılımı
      4.1.3.4. Suç Sınıflandırması (wcisclass)
   4.1.4. Korelasyon Analizi
   4.1.5. Çok Değişkenli İlişkiler ve İleri Düzey Analizler

### 4.2. Model Performans Sonuçları
   4.2.1. Baseline Model Performansı
   4.2.2. Kategori Optimizasyon Denemeleri
      4.2.2.1. 4-Kategorili Sistem
      4.2.2.2. BALANCED 3-Kategorili Sistem
   4.2.3. Hyperparameter Tuning ve Feature Selection Sonuçları
   4.2.4. Ensemble Model Performansı
      4.2.4.1. XGBoost vs LightGBM Karşılaştırması
      4.2.4.2. Simple vs Weighted Average
      4.2.4.3. Final Model Performansı
   4.2.5. Kategori Bazlı Performans Analizi

### 4.3. Feature Importance ve SHAP Analizi
   4.3.1. XGBoost Feature Importance Sonuçları
   4.3.2. SHAP Summary Plot Analizi
   4.3.3. SHAP Dependence Plot Analizleri
      4.3.3.1. Suç Şiddeti (highest_severity)
      4.3.3.2. Yaş Etkisi (age_offense)
      4.3.3.3. Sosyoekonomik Faktörler (pct_black)
      4.3.3.4. Sabıka Geçmişi (max_hist_jail)
   4.3.4. SHAP Force Plot ve Vaka İncelemesi

### 4.4. Fairness ve Bias Değerlendirmesi
   4.4.1. Demographic Parity Metrikleri
   4.4.2. Irk Bazlı Fairness Analizi
   4.4.3. Cinsiyet Bazlı Fairness Analizi
   4.4.4. Fairness-Accuracy Trade-off Değerlendirmesi
   4.4.5. Dolaylı Bias Tespiti

### 4.5. Literatür ile Karşılaştırma ve Tartışma
   4.5.1. Performans Karşılaştırması (Önceki Çalışmalar)
   4.5.2. Metodolojik Farklılıklar
   4.5.3. Fairness Açısından Karşılaştırma
   4.5.4. Bulguların Değerlendirilmesi

---

## 5. SONUÇ VE ÖNERİLER

### 5.1. Genel Değerlendirme ve Ana Bulgular
   5.1.1. Model Performansı Özeti
   5.1.2. Fairness ve Adalet Bulguları
   5.1.3. Kategori Optimizasyonu Başarısı

### 5.2. Teorik ve Pratik Katkılar
   5.2.1. Teorik Katkılar
      5.2.1.1. Kategori Optimizasyon Metodolojisi
      5.2.1.2. Fairness-Accuracy Dengesi
      5.2.1.3. SHAP ile Bias Tespiti
   5.2.2. Pratik Katkılar
      5.2.2.1. Yargı Karar Destek Sistemi
      5.2.2.2. Ceza Tutarlılığı
      5.2.2.3. Kaynak Tahsisi Optimizasyonu

### 5.3. Kısıtlamalar
   5.3.1. Veri Seti Kısıtlamaları
   5.3.2. Metodolojik Kısıtlamalar
   5.3.3. Genellenebilirlik Sınırlamaları

### 5.4. Gelecek Çalışmalar için Öneriler
   5.4.1. Multi-State Analiz ve Genişletme
   5.4.2. Temporal (Zamansal) Analiz
   5.4.3. Fairness-Aware Learning Algoritmaları
   5.4.4. Deep Learning Yaklaşımları
   5.4.5. Causal Inference (Nedensel Çıkarım)
   5.4.6. Recidivism (Tekerrür) Entegrasyonu
   5.4.7. Gelişmiş Explainable AI Teknikleri
   5.4.8. Real-Time Deployment ve Uygulama
