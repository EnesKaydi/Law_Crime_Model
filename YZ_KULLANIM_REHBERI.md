# YZ İÇİN TEZ YAZIM REHBERİ

> **Bu dosyayı bir YZ asistanına (ChatGPT, Claude, vb.) vererek, hazırladığım bulguları tez formatına dönüştürebilirsin.**

---

## 📚 MEVCUT DOSYALAR VE İÇERİKLERİ

Elimde **4 ana doküman** var:

### 1️⃣ TEZ_METODOLOJI.md (1,100+ satır)
**İçerik:**
- Giriş ve Literatür Taraması
- Veri Seti ve Ön İşleme Detayları
- Keşifsel Veri Analizi (EDA) Metodolojisi  
- Özellik Mühendisliği (Feature Engineering)
- Model Geliştirme Süreci
- Final Model Mimarisi (Ensemble)

**Kullanım:** Tezin "Yöntem/Metodoloji" bölümü için

---

### 2️⃣ TEZ_BULGULAR_1.md (1,000+ satır)
**İçerik:**
- Veri Seti Genel Karakteristikleri (Tablo 1.1)
- Hedef Değişken (Jail) Dağılımı ve İstatistikleri (Tablo 1.2-1.4, Grafik 1.1-1.3)
- Demografik Değişkenlerin Dağılımı:
  - Cinsiyet Analizi (Tablo 1.5, Grafik 1.4)
  - Irk/Etnik Köken ve Sistemik Bias (Tablo 1.6, Grafik 1.5)
  - Dava Türü Dağılımı (Tablo 1.7, Grafik 1.6)
  - Suç Türleri Analizi (Tablo 1.8, Grafik 1.7)
- Korelasyon Analizi (Tablo 1.9-1.11, Grafik 1.8-1.9)
- Baseline Model Performansı (Tablo 2.1-2.3)

**Kullanım:** Tezin "Bulgular" bölümünün 1. kısmı

---

### 3️⃣ TEZ_BULGULAR_2.md (1,000+ satır)
**İçerik:**
- Kategori Optimizasyon Süreci:
  - 4 Kategorili Sistem Denemesi (Tablo 2.4)
  - BALANCED 3 Kategori Sistemi - BAŞARI (Tablo 2.5-2.7, Grafik 2.3-2.4)
- Ensemble Model Performansı (Tablo 2.8-2.10, Grafik 2.5-2.6)
- Feature Importance Analizi (Tablo 2.11, Grafik 2.7)
- SHAP Analizi:
  - Summary Plot (Tablo 2.12, Grafik 2.8)
  - Dependence Plot'lar (Grafik 2.9-2.12)
  - Force Plot Örneği (Grafik 2.13)

**Kullanım:** Tezin "Bulgular" bölümünün 2. kısmı

---

### 4️⃣ TEZ_BULGULAR_3.md (1,000+ satır)
**İçerik:**
- SHAP Analizi Devamı (Yaş, Sosyoekonomik Bias)
- Feature Selection Denemesi - BAŞARISIZ (Tablo 3.1)
- Fairness ve Bias Değerlendirmesi:
  - Irk Bazlı Fairness (Tablo 4.1, Grafik 4.1)
  - Cinsiyet Bazlı Fairness (Tablo 4.2, Grafik 4.2)
  - Fairness-Accuracy Trade-off (Tablo 4.3, Grafik 4.3)
- Sonuç ve Tartışma:
  - Ana Bulgular Özeti (Tablo 5.1-5.4)
  - Literatür Karşılaştırması (detaylı)
  - Teorik ve Pratik Katkılar
  - Kısıtlamalar
  - Gelecek Çalışmalar (8 öneri)
- Ekler (Grafik ve Tablo Listesi)

**Kullanım:** Tezin "Bulgular" 3. kısmı, "Tartışma" ve "Sonuç" bölümleri

---

## 🎯 YZ'YE VERECEĞİN PROMPT ÖRNEĞİ

```
Merhaba! Makine öğrenmesi ile ceza tahmini üzerine bir yüksek lisans tezi yazıyorum. 
Elimde detaylı bulgular ve metodoloji var. 

Tezimin yapısı şöyle:
1. GİRİŞ
2. LİTERATÜR TARAMASI
3. METODOLOJİ
   3.1. Veri Seti
   3.2. Veri Ön İşleme
   3.3. Keşifsel Veri Analizi
   3.4. Özellik Mühendisliği
   3.5. Model Geliştirme
4. BULGULAR
   4.1. Veri Analizi Bulguları
   4.2. Model Performans Sonuçları
   4.3. Feature Importance ve Açıklanabilirlik
   4.4. Fairness Değerlendirmesi
5. TARTIŞMA
6. SONUÇ VE ÖNERİLER

Sana 4 dosya vereceğim:
- TEZ_METODOLOJI.md → Bölüm 3 için
- TEZ_BULGULAR_1.md → Bölüm 4.1 için
- TEZ_BULGULAR_2.md → Bölüm 4.2-4.3 için
- TEZ_BULGULAR_3.md → Bölüm 4.4, 5 ve 6 için

GÖREV:
[Burada hangi bölümü istediğini belirt, örneğin:]

"Bölüm 3.1 (Veri Seti) için TEZ_METODOLOJI.md dosyasındaki 'Veri Seti ve Ön İşleme' 
kısmını akademik tez diline çevir. Tablolar ve grafikler olduğu gibi kalsın, 
sadece metin kısımlarını yüksek lisans tezi standardında yaz."
```

---

## 📋 BÖLÜM BÖLÜM KULLANIM REHBERİ

### BÖLÜM 1: GİRİŞ
**Kullanılacak Dosya:** TEZ_METODOLOJI.md  
**İlgili Kısım:** "1. GİRİŞ VE LİTERATÜR TARAMASI" → "1.1. Araştırmanın Amacı ve Önemi"

**YZ'ye Prompt:**
```
TEZ_METODOLOJI.md dosyasından "1.1. Araştırmanın Amacı ve Önemi" kısmını kullanarak,
tezimin GİRİŞ bölümünü yaz. 

Şunları ekle:
- Çalışmanın önemi
- Araştırma soruları
- Tezin kapsamı
- Bölüm organizasyonu (hangi bölümde ne var)

Akademik, formal dil kullan. 2-3 sayfa olsun.
```

---

### BÖLÜM 2: LİTERATÜR TARAMASI
**Kullanılacak Dosya:** TEZ_METODOLOJI.md + TEZ_BULGULAR_3.md  
**İlgili Kısımlar:** 
- TEZ_METODOLOJI.md → "1.2. Literatür Özeti"
- TEZ_BULGULAR_3.md → "Tablo 5.4: Detaylı Literatür Karşılaştırması"

**YZ'ye Prompt:**
```
TEZ_METODOLOJI.md'deki "1.2. Literatür Özeti" ve TEZ_BULGULAR_3.md'deki 
"Tablo 5.4" kullanarak LİTERATÜR TARAMASI bölümünü yaz.

Şunları dahil et:
- Ceza tahmini alanında yapılan çalışmalar (kronolojik)
- Kullanılan yöntemler (Logistic Regression → XGBoost → Ensemble)
- Performans karşılaştırmaları (Tablo 5.4)
- Bu çalışmanın araştırma boşluğunu nasıl doldurduğu
- Fairness ve bias çalışmaları

4-5 sayfa, akademik üslup, APA formatında.
```

---

### BÖLÜM 3.1: VERİ SETİ
**Kullanılacak Dosya:** TEZ_METODOLOJI.md + TEZ_BULGULAR_1.md  
**İlgili Kısımlar:**
- TEZ_METODOLOJI.md → "2.1. Veri Kaynağı", "2.2. Veri Yapısı"
- TEZ_BULGULAR_1.md → "Tablo 1.1: Veri Seti Özet İstatistikleri"

**YZ'ye Prompt:**
```
TEZ_METODOLOJI.md'deki Bölüm 2.1-2.2 ve TEZ_BULGULAR_1.md'deki Tablo 1.1 
kullanarak "Veri Seti" alt bölümünü yaz.

İçerik:
- Veri kaynağı (Wisconsin Circuit Court Database)
- Zaman aralığı (2013-2015)
- Veri büyüklüğü (525,379 kayıt, 54 özellik)
- Değişken kategorileri (Tablo 1.1 referans)
- Veri toplama süreci

2-3 sayfa, tablo referanslarını koru.
```

---

### BÖLÜM 3.2: VERİ ÖN İŞLEME
**Kullanılacak Dosya:** TEZ_METODOLOJI.md  
**İlgili Kısım:** "2.3. Veri Temizleme ve Ön İşleme Süreci"

**YZ'ye Prompt:**
```
TEZ_METODOLOJI.md'deki "2.3. Veri Temizleme" kullanarak VERİ ÖN İŞLEME 
bölümünü yaz.

Adımlar:
1. Eksik değer analizi ve stratejileri
2. Aykırı değer (outlier) analizi → NEDEN KORUNDU
3. Kategorik değişken kodlama (Label, One-Hot, Target Encoding)
4. Özellik ölçeklendirme (StandardScaler)
5. Train-Test bölünmesi (80/20, stratified)

Kod snippet'leri varsa Python formatında ekle.
3-4 sayfa.
```

---

### BÖLÜM 3.3: KEŞİFSEL VERİ ANALİZİ
**Kullanılacak Dosya:** TEZ_BULGULAR_1.md  
**İlgili Kısım:** "1. KEŞİFSEL VERİ ANALİZİ BULGULARI" (tamamı)

**YZ'ye Prompt:**
```
TEZ_BULGULAR_1.md'deki tüm Bölüm 1 kullanarak KEŞİFSEL VERİ ANALİZİ 
bölümünü yaz.

Alt başlıklar:
3.3.1. Hedef Değişken Analizi (Tablo 1.2, Grafik 1.1-1.2)
3.3.2. Demografik Değişkenler (Tablo 1.5-1.8, Grafik 1.4-1.7)
3.3.3. Korelasyon Analizi (Tablo 1.9-1.11, Grafik 1.8-1.9)
3.3.4. Sistemik Bias Tespiti (ÖNEMLİ!)

Grafik ve tablo referanslarını koru.
Sistemik bias bulgularını vurgula (African American 3.76x aşırı temsil).
5-6 sayfa.
```

---

### BÖLÜM 3.4: ÖZELLİK MÜHENDİSLİĞİ
**Kullanılacak Dosya:** TEZ_METODOLOJI.md  
**İlgili Kısım:** "4. ÖZELLİK MÜHENDİSLİĞİ"

**YZ'ye Prompt:**
```
TEZ_METODOLOJI.md Bölüm 4 kullanarak ÖZELLİK MÜHENDİSLİĞİ bölümünü yaz.

İçerik:
- Kategorik değişken kodlama teknikleri (detaylı)
- Target encoding nasıl yapıldı (wcisclass için)
- Multicollinearity problemi ve çözümü
- Final özellik seti (41 özellik)

2-3 sayfa.
```

---

### BÖLÜM 3.5: MODEL GELİŞTİRME
**Kullanılacak Dosya:** TEZ_METODOLOJI.md  
**İlgili Kısım:** "5. MODEL GELİŞTİRME SÜRECİ", "6. FİNAL MODEL"

**YZ'ye Prompt:**
```
TEZ_METODOLOJI.md Bölüm 5-6 kullanarak MODEL GELİŞTİRME bölümünü yaz.

Alt başlıklar:
3.5.1. Model Seçimi (Neden XGBoost/LightGBM?)
3.5.2. Baseline Model
3.5.3. Kategori Optimizasyonu (KRITIK ADIM!)
3.5.4. Hyperparameter Tuning
3.5.5. Ensemble Model Tasarımı

Kategori optimizasyonunu detaylandır (BALANCED sistem breakthrough!).
4-5 sayfa.
```

---

### BÖLÜM 4.1: VERİ ANALİZİ BULGULARI
**Kullanılacak Dosya:** TEZ_BULGULAR_1.md  
**İlgili Kısım:** Bölüm 1 (tekrar, ama bu sefer BULGULAR bölümü olarak)

**YZ'ye Prompt:**
```
TEZ_BULGULAR_1.md Bölüm 1 kullanarak VERİ ANALİZİ BULGULARI bölümünü yaz.

NOT: Bu, metodolojide EDA olarak geçti. Şimdi BULGULAR olarak sunulacak.
Daha çok bulgulara ve istatistiksel anlamlılığa odaklan.

Kritik bulgular:
- Jail dağılımı aşırı sağa çarpık (Skewness: +15.8)
- African American 3.76x aşırı temsil (SİSTEMİK BİAS!)
- OWI (alkol) en yaygın suç (%23.6)
- highest_severity en güçlü korelasyon (r=0.31)

Tablo ve grafikleri referans göster.
4-5 sayfa.
```

---

### BÖLÜM 4.2: MODEL PERFORMANS SONUÇLARI
**Kullanılacak Dosya:** TEZ_BULGULAR_1.md + TEZ_BULGULAR_2.md  
**İlgili Kısımlar:**
- TEZ_BULGULAR_1.md → "2. MODEL PERFORMANS SONUÇLARI"
- TEZ_BULGULAR_2.md → Kategori optimizasyon, Ensemble

**YZ'ye Prompt:**
```
TEZ_BULGULAR_1.md Bölüm 2 ve TEZ_BULGULAR_2.md kullanarak 
MODEL PERFORMANS SONUÇLARI bölümünü yaz.

Alt başlıklar:
4.2.1. Baseline Model (R²=0.44, YETERSIZ)
4.2.2. Kategori Optimizasyon Denemeleri
   - 4 Kategori: BAŞARISIZ
   - BALANCED 3 Kategori: BAŞARILI! (+42.5% R²)
4.2.3. Ensemble Model (FINAL)
   - XGBoost vs LightGBM
   - Simple vs Weighted Average
   - Final: R²=0.6321 (+43.5% toplam iyileşme!)
4.2.4. Literatür Karşılaştırması

Tablo 2.5, 2.7, 2.8, 2.9 önemli!
5-6 sayfa.
```

---

### BÖLÜM 4.3: FEATURE IMPORTANCE VE AÇIKLANABİLİRLİK
**Kullanılacak Dosya:** TEZ_BULGULAR_2.md  
**İlgili Kısım:** "2.4. Feature Importance"

**YZ'ye Prompt:**
```
TEZ_BULGULAR_2.md Bölüm 2.4 kullanarak FEATURE IMPORTANCE VE 
AÇIKLANABİLİRLİK bölümünü yaz.

Alt başlıklar:
4.3.1. XGBoost Feature Importance (Tablo 2.11, Grafik 2.7)
   - highest_severity dominant (%28.47)
   - Top 5 özellik %67 importance
4.3.2. SHAP Analizi (Grafik 2.8-2.13)
   - Summary Plot
   - Dependence Plot'lar (severity, age, pct_black, max_hist_jail)
   - Force Plot örneği (Vaka #12,543)
4.3.3. Sosyoekonomik Bias SHAP Bulguları
   - pct_black dolaylı bias (Mean SHAP: +8.2 gün)

Grafik referansları kritik!
4-5 sayfa.
```

---

### BÖLÜM 4.4: FAİRNESS DEĞERLENDİRMESİ
**Kullanılacak Dosya:** TEZ_BULGULAR_3.md  
**İlgili Kısım:** "4. FAİRNESS VE BİAS DEĞERLENDİRMESİ"

**YZ'ye Prompt:**
```
TEZ_BULGULAR_3.md Bölüm 4 kullanarak FAİRNESS DEĞERLENDİRMESİ bölümünü yaz.

Alt başlıklar:
4.4.1. Demographic Parity Metrikleri
4.4.2. Irk Bazlı Fairness (Tablo 4.1, Grafik 4.1)
   - Fairness Ratio: 0.987 (Adil ✅)
4.4.3. Cinsiyet Bazlı Fairness (Tablo 4.2, Grafik 4.2)
   - Fairness Ratio: 0.960 (Adil ✅)
4.4.4. Fairness-Accuracy Trade-off (Grafik 4.3)

VURGU: Model, sistemik bias'a rağmen adil tahminler yapıyor!
3-4 sayfa.
```

---

### BÖLÜM 5: TARTIŞMA
**Kullanılacak Dosya:** TEZ_BULGULAR_3.md  
**İlgili Kısım:** "5. SONUÇ VE TARTIŞMA" (5.1-5.4)

**YZ'ye Prompt:**
```
TEZ_BULGULAR_3.md Bölüm 5.1-5.4 kullanarak TARTIŞMA bölümünü yaz.

Alt başlıklar:
5.1. Bulguların Özeti
5.2. Literatür ile Karşılaştırma (Tablo 5.4)
   - %31-124 daha iyi performans!
5.3. Teorik Katkılar
   - Kategori optimizasyon metodolojisi
   - Fairness-accuracy dengesinin mümkün olduğu
5.4. Pratik Katkılar
   - Yargı desteği
   - Ceza tutarlılığı
   - Kaynak tahsisi
5.5. Kısıtlamalar
   - Tek eyalet, zaman kısıtı, dolaylı bias

4-5 sayfa, eleştirel bakış.
```

---

### BÖLÜM 6: SONUÇ VE ÖNERİLER
**Kullanılacak Dosya:** TEZ_BULGULAR_3.md  
**İlgili Kısım:** "5.5. Gelecek Çalışmalar", "6. SONUÇ"

**YZ'ye Prompt:**
```
TEZ_BULGULAR_3.md Bölüm 5.5 ve 6 kullanarak SONUÇ VE ÖNERİLER bölümünü yaz.

6.1. Genel Değerlendirme
   - R²=0.6321 (+43.5% iyileşme)
   - Fairness kabul edilebilir (0.95+)
   - Literatürün üzerinde

6.2. Ana Katkılar (5 madde)

6.3. Gelecek Çalışmalar (8 öneri):
   1. Multi-state genişletme
   2. Temporal analysis
   3. Fairness-aware learning
   4. Deep learning
   5. Causal inference
   6. Recidivism entegrasyonu
   7. Explainable AI
   8. Real-time deployment

6.4. Final Mesaj (etik vurgu)

3-4 sayfa, pozitif ama dikkatli ton.
```

---

## 🎨 ÖZEL İSTEKLER İÇİN PROMPTlar

### Tablo Oluşturma:
```
TEZ_BULGULAR_1.md'deki "Tablo 1.6: Irk Dağılımı" tablosunu LaTeX formatında yaz.
```

### Grafik Açıklaması:
```
"Grafik 2.8: SHAP Summary Plot" için detaylı açıklama yaz. 
Grafikte ne gösteriliyor, nasıl yorumlanır, okuyucu ne anlamalı?
2-3 paragraf.
```

### İstatistiksel Test Ekleme:
```
TEZ_BULGULAR_1.md'deki "African American aşırı temsil" bulgusuna 
Chi-square testi ekle. Hipotez, test istatistiği, p-değeri, karar.
```

---

## ✅ KONTROL LİSTESİ

Tez yazarken şunları kontrol et:

- [ ] Tüm tablolar numaralandırıldı mı? (Tablo 1.1, 1.2, ...)
- [ ] Tüm grafikler referans gösterildi mi? (Şekil 1.1, Grafik 2.3, ...)
- [ ] İstatistiksel testler eklendi mi? (p-değerleri, CI'lar)
- [ ] Literatür atıfları yapıldı mı? (Dressel & Farid, 2018, vb.)
- [ ] Kısaltmalar ilk kullanımda açıldı mı? (EDA, SHAP, RMSE)
- [ ] Akademik dil tutarlı mı? (1. tekil şahıs YOK, "Bu çalışmada..." kullan)
- [ ] Bölüm geçişleri akıcı mı?
- [ ] Etik vurgu yapıldı mı? (Özellikle fairness bölümünde)

---

## 🚀 HIZLI BAŞLANGIÇ ÖRNEĞİ

**Senaryo:** Sadece "Bulgular" bölümünü yazdırmak istiyorsun.

**Adım 1:** YZ'ye şunu söyle:
```
3 dosyam var: TEZ_BULGULAR_1.md, TEZ_BULGULAR_2.md, TEZ_BULGULAR_3.md

Bunları kullanarak tezimin BULGULAR bölümünü yaz.

Yapı:
4. BULGULAR
   4.1. Veri Analizi Bulguları (TEZ_BULGULAR_1.md)
   4.2. Model Performans Sonuçları (TEZ_BULGULAR_2.md)
   4.3. Feature Importance (TEZ_BULGULAR_2.md)
   4.4. Fairness Değerlendirmesi (TEZ_BULGULAR_3.md)

Akademik üslup, tablo/grafik referansları koru, 15-20 sayfa olsun.
```

**Adım 2:** 3 dosyayı sırayla yapıştır.

**Adım 3:** "Devam et" de, tamamlat.

**Adım 4:** Çıktıyı al, Word'e yapıştır, biçimlendir. Bitti! ✅

---

## 💡 İPUÇLARI

1. **YZ'ye tüm dosyayı birden verme:** Bölüm bölüm işle (token limiti nedeniyle)
2. **Tablo/grafik referanslarını kontrol et:** YZ bazen değiştirebilir
3. **İstatistiksel değerleri doğrula:** Sayılar doğru kopyalandı mı?
4. **Akademik dil kontrolü:** "biz", "ben" yerine "bu çalışmada" kullandır
5. **Tutarlılık:** Aynı kavram için aynı terim kullan (örn: "jail" → "hapis cezası süresi")

---

**📌 NOT:** Bu rehber, hazırladığım 4 dosyayı maksimum verimle tez formatına dönüştürmen için tasarlandı. Herhangi bir YZ asistanına (ChatGPT, Claude, Gemini) bu rehberi + ilgili dosyayı vererek, tez bölümlerini otomatik yazdırabilirsin!

**🎓 Başarılar! Tez yazımı bu şekilde çok daha hızlı olacak!** ✨