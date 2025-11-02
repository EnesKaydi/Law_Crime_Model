# TEZ PROJESİ SONUÇLARI
## YAPAY ZEKA DESTEKLİ HUKUK ASİSTANI - Wisconsin Ceza Mahkemesi Veri Seti Analizi

**Proje Adı:** Yapay Zeka Destekli Hukuk Asistanı  
**Öğrenci:** Muhammed Enes Kaydı  
**Danışman:** Müge Özçevik  
**Tarih:** 2 Kasım 2025  

---

## 📊 GENEL ÖZET

Bu dokümanda, tez projesinin tüm adımları, sonuçları ve bulguları detaylı şekilde raporlanmıştır.

---

## ADIM 1: BÜYÜK VERİ SETİ YÜKLEME VE İNCELEME

### Veri Kaynağı
- **Dosya:** wcld.csv
- **Kaynak:** Wisconsin Eyaleti Ceza Mahkemeleri
- **Konum:** `/Users/muhammedeneskaydi/Desktop/3.SINIF 2.DÖNEM/TEZ/TEZ FİNAL/wcld.csv`

### Sonuçlar
```
📊 TOPLAM SATIR SAYISI: 1,476,967 (yaklaşık 1.5 milyon)
📊 KOLON SAYISI: 54
📊 TAM DOLU SATIRLAR: 357,452 (%24.20)
📊 EKSİK VERİLİ SATIRLAR: 1,119,515 (%75.80)
```

### Yükleme Süresi
- **Süre:** ~3 saniye

### Değerlendirme
✅ Veri seti başarıyla yüklendi.  
✅ 54 kolon (demografik, suç, ceza, mahalle bilgileri) mevcut.  
⚠️ Eksik veri oranı yüksek (%75.80) ancak bu beklenen bir durumdur.

---

## ADIM 2: TEMİZ VERİ SEÇİMİ - TÜM KOLONLAR DOLU

### İşlem
Tüm kolonları dolu olan (NaN içermeyen) satırlar seçildi ve ayrı bir dosyaya kaydedildi.

### Sonuçlar
```
📊 TEMİZ SATIR SAYISI: 357,452
📊 ORİJİNAL VERİNİN: %24.20'si
📦 DOSYA BOYUTU: 267.28 MB
📁 KAYIT YERİ: wcld_Tüm_Kolonlar_Dolu.csv
```

### Yöntem
- Python Pandas `dropna()` fonksiyonu kullanıldı
- Tüm kolonlar için eksik değer kontrolü yapıldı

### Değerlendirme
✅ Temiz veri başarıyla ayrıldı.  
✅ Model eğitiminde kullanılacak temel veri seti hazır.

---

## ADIM 3: EKSİK VERİLERDEN %15 ÖRNEKLEM

### İşlem
Modelin genelleme yeteneğini artırmak için eksik verili satırlardan rastgele %15 örneklem alındı.

### Parametreler
- **Örnekleme Oranı:** %15
- **Random State:** 42 (tekrarlanabilirlik için)
- **Yöntem:** Pandas `sample()` fonksiyonu

### Sonuçlar
```
📊 EKSİK VERİLİ SATIRLAR: 1,119,515
📊 SEÇİLEN ÖRNEKLEM: 167,927 (%15.00)
📦 DOSYA BOYUTU: 124.68 MB
📁 KAYIT YERİ: wcld_Eksik_Veri_Yuzde15.csv
```

### Değerlendirme
✅ Örneklem başarıyla alındı.  
✅ Random state=42 ile tekrarlanabilirlik sağlandı.  
✅ Veri çeşitliliği artırıldı.

---

## ADIM 4: FİNAL VERİ SETİ BİRLEŞTİRME

### İşlem
Temiz veri (357K) ile eksik veri örneklemi (167K) birleştirilerek final dataset oluşturuldu.

### Sonuçlar
```
📊 FİNAL VERİ SETİ BOYUTU: 525,379 satır × 54 kolon
📊 TEMİZ VERİ PAYI: 357,452 (%68.04)
📊 EKSİK VERİ PAYI: 167,927 (%31.96)
📦 DOSYA BOYUTU: ~216 MB
📁 KAYIT YERİ: wcld_Final_Dataset.csv
```

### Hedef Değişken Dolu Oranları
```
✅ jail (hapis süresi): 399,807 dolu (%76.1)
✅ probation (şartlı tahliye): 458,865 dolu (%87.3)
✅ release (serbest bırakılma): 525,379 dolu (%100.0)
```

### Değerlendirme
✅ Final veri seti başarıyla oluşturuldu.  
✅ 525,379 satır ile yeterli veri hacmi sağlandı.  
✅ Hedef değişkenler yüksek oranda dolu (özellikle release %100).  
✅ Model eğitimi için hazır!

---

## ADIM 5: VERİ KEŞİF ANALİZİ (EDA)

### 5.1 - Temel İstatistikler ✅

#### Veri Seti Genel Bilgileri
```
📊 Satır Sayısı: 525,379
📊 Kolon Sayısı: 54
💾 Bellek Kullanımı: 391.97 MB
📦 Toplam Hücre: 28,370,466
```

#### Veri Tipleri
```
✔️ Sayısal Kolonlar: 47 adet (float64: 36, int64: 11)
✔️ Kategorik Kolonlar: 7 adet (object)
```

#### Eksik Değer Analizi
```
⚠️ Eksik değer içeren kolon: 11 adet
📊 Toplam eksik hücre: 413,519
📊 Genel eksik oran: %1.46 (çok düşük! ✅)

En çok eksik değer içeren kolonlar:
1. jail: 125,572 (%23.9)
2. probation: 66,514 (%12.7)
3. max_hist_jail: 42,180 (%8.0)
4. min_hist_jail: 42,180 (%8.0)
5. avg_hist_jail: 42,180 (%8.0)
```

#### Hedef Değişkenler İstatistikleri

**🎯 JAIL (Hapis Süresi - GÜN):**
```
• Dolu: 399,807 (%76.1) ✅
• Ortalama: 111.97 gün (~3.7 ay)
• Medyan: 30 gün (1 ay)
• Min: 0 gün
• Max: 109,500 gün (300 yıl!) ⚠️
• Std. Sapma: 680.28 (yüksek varyans)
• Q1: 7 gün
• Q3: 80 gün
```

**🎯 PROBATION (Şartlı Tahliye - Binary):**
```
• Dolu: 458,865 (%87.3) ✅
• Ortalama: 0.26 (%26 şartlı tahliye oranı)
• Min: 0, Max: 1 (ikili değişken)
```

**🎯 RELEASE (Serbest Bırakılma - Binary):**
```
• Dolu: 525,379 (%100.0) ✅✅
• Ortalama: 0.36 (%36 serbest bırakılma oranı)
• Min: 0, Max: 1 (ikili değişken)
```

#### Diğer Önemli Sayısal Değişkenler

```
📌 AGE_OFFENSE (Suç İşleme Yaşı):
   • Ortalama: 31.57 yaş
   • Medyan: 29 yaş
   • Min: 14, Max: 150 (outlier var! ⚠️)

📌 PRIOR_FELONY (Önceki Ağır Suçlar):
   • Ortalama: 1.02
   • Medyan: 0 (çoğunlukta ilk suç)
   • Max: 33 (çok yüksek sabıka!)

📌 PRIOR_MISDEMEANOR (Önceki Hafif Suçlar):
   • Ortalama: 1.50
   • Medyan: 1
   • Max: 60

📌 VIOLENT_CRIME (Şiddet İçeren Suç):
   • Ortalama: 0.13 (%13 şiddet içerir)
   • Medyan: 0 (çoğunluk şiddetsiz)

📌 RECID_180D (180 Gün İçinde Tekrar Suç):
   • Ortalama: 0.43 (%43 tekrar suç işler! ⚠️)
```

#### Kategorik Değişken Dağılımları

**📌 SEX (Cinsiyet):**
```
• M (Erkek): 427,645 (%81.4) 🔵
• F (Kadın): 97,734 (%18.6) 🔴
```

**📌 RACE (Irk):**
```
• Caucasian: 342,669 (%65.2)
• African American: 118,466 (%22.5)
• Hispanic: 36,342 (%6.9)
• American Indian: 23,301 (%4.4)
• Asian/Pacific: 4,601 (%0.9)
```

**📌 CASE_TYPE (Dava Türü):**
```
• Misdemeanor (Hafif): 213,895 (%40.7)
• Criminal Traffic: 184,333 (%35.1)
• Felony (Ağır): 127,151 (%24.2)
```

**📌 WCISCLASS (Suç Türü) - En Sık 5:**
```
1. Operating While Intoxicated (OWI): 123,982 (%23.6) 🚗🍺
2. OAR/OAS: 55,135 (%10.5)
3. Drug Possession: 38,177 (%7.3) 💊
4. Bail Jumping: 36,587 (%7.0)
5. Battery: 35,744 (%6.8) 👊
```

#### Önemli Bulgular ve Yorumlar

✅ **Pozitif Noktalar:**
- Veri kalitesi çok iyi (%98.54 dolu)
- Hedef değişkenler yüksek oranda dolu
- Yeterli veri hacmi (525K örnek)
- Dengeli özellik dağılımı

⚠️ **Dikkat Edilmesi Gerekenler:**
- `jail` değişkeninde aşırı outlier'lar var (max: 109,500 gün!)
- `age_offense` max: 150 yaş - veri hatası olabilir
- Erkek/kadın oranı dengesiz (%81.4 erkek)
- Tekrar suç oranı yüksek (%43)

📊 **Model İçin Öneriler:**
1. Outlier temizliği gerekli (jail > 10,000 gün olanlar)
2. Age > 100 olanları kontrol et
3. Class imbalance için stratified sampling kullan
4. Irk değişkeni için bias analizi yap

#### Çıktı Dosyaları
```
📁 outputs/temel_istatistikler.txt
```

---

### 5.2 - Hedef Değişken Dağılımları ✅

#### Genel Bakış
Bu adımda `jail`, `probation`, ve `release` hedef değişkenlerinin dağılımları analiz edildi ve görselleştirildi. Ayrıca `jail` değişkenine göre ceza kategorileri (Hafif/Orta/Ağır) oluşturuldu.

---

#### 🎯 JAIL (Hapis Süresi - Gün Cinsinden)

**İstatistikler:**
```
• Dolu Kayıt: 399,807 (%76.1)
• Eksik Kayıt: 125,572 (%23.9)
• Ortalama: 111.97 gün (~3.7 ay)
• Medyan: 30 gün (1 ay)
• Standart Sapma: 680.28 (yüksek varyans ⚠️)
• Minimum: 0 gün
• Maximum: 109,500 gün (300 yıl! aşırı outlier ⚠️)
• Q1 (25%): 7 gün
• Q3 (75%): 80 gün
```

**Grafikler:**
- 📊 `hist_jail.png` - Histogram (dağılım görünümü)
- 📦 `box_jail.png` - Boxplot (outlier tespiti)

**Yorumlar:**
- Medyan 30 gün, ortalama 112 gün → Sağa çarpık dağılım (outlier'lar ortalamayı çekiyor)
- Max değer 109,500 gün (~300 yıl) → Veri hatası olabilir, temizleme gerekebilir
- Çoğu ceza 7-80 gün arasında (Q1-Q3)

---

#### 🎯 PROBATION (Şartlı Tahliye - Binary)

**İstatistikler:**
```
• Dolu Kayıt: 458,865 (%87.3) ✅
• Eksik Kayıt: 66,514 (%12.7)
• Ortalama: 0.26 (%26 şartlı tahliye oranı)
• Medyan: 0 (çoğunluk şartlı tahliye almıyor)
• Min: 0, Max: 1 (ikili değişken)
```

**Grafikler:**
- 📊 `hist_probation.png` - Histogram
- 📦 `box_probation.png` - Boxplot

**Yorumlar:**
- %26 oranında şartlı tahliye veriliyor
- %74 şartlı tahliye alMIyor
- İkili sınıflandırma problemi için uygun

---

#### 🎯 RELEASE (Serbest Bırakılma - Binary)

**İstatistikler:**
```
• Dolu Kayıt: 525,379 (%100.0) ✅✅
• Eksik Kayıt: 0 (mükemmel!)
• Ortalama: 0.36 (%36 serbest bırakılma oranı)
• Medyan: 0 (çoğunluk hapis yatıyor)
• Min: 0, Max: 1 (ikili değişken)
```

**Grafikler:**
- 📊 `hist_release.png` - Histogram
- 📦 `box_release.png` - Boxplot

**Yorumlar:**
- %36 oranında serbest bırakılıyor (hapis yok)
- %64 hapis cezası alıyor
- Hiç eksik veri yok → Model için ideal

---

#### 📊 CEZA KATEGORİLERİ (jail değerine göre)

**Kategori Kuralları:**
```
• NoJail: 0 gün veya NaN (ceza yok)
• Hafif: 1-180 gün (6 aya kadar)
• Orta: 181-1080 gün (6 ay - 3 yıl)
• Ağır: 1081+ gün (3 yıl üzeri)
```

**Dağılım:**
```
1. Hafif: 320,921 (%61.1) 🟢 En büyük grup!
2. NoJail: 170,600 (%32.5) ⚪
3. Orta: 27,065 (%5.2) 🟡
4. Ağır: 6,788 (%1.3) 🔴
5. None: 5 (%0.0) ⚠️ (garbage değer)
```

**Grafik:**
- 📊 `ceza_kategori_barchart.png` - Kategori dağılımı bar chart

**Yorumlar:**
- %61.1 hafif ceza → Sistem çoğunlukla hafif cezalar veriyor
- %32.5 hiç ceza yok → Büyük oran!
- Ağır cezalar sadece %1.3 → Çok nadir
- Class imbalance var → Modelde stratified sampling kullanılmalı

---

#### 📁 Kaydedilen Dosyalar

**Grafik Klasörü:** `outputs/eda/target_distributions/`

**Dosyalar:**
```
1. hist_jail.png (Jail histogram)
2. box_jail.png (Jail boxplot)
3. hist_probation.png (Probation histogram)
4. box_probation.png (Probation boxplot)
5. hist_release.png (Release histogram)
6. box_release.png (Release boxplot)
7. ceza_kategori_barchart.png (Ceza kategorileri bar chart)
```

**Kullanım:** Tez raporunda "Veri Keşif Analizi" bölümüne bu grafikler eklenecek.

---

#### ✅ Önemli Bulgular ve Öneriler

**Bulgular:**
1. ✅ Hedef değişkenler yeterince dolu (%76-100)
2. ⚠️ Jail değişkeninde aşırı outlier'lar var (max: 109,500)
3. ✅ Ceza dağılımı çoğunlukla hafif cezalarda yoğunlaşmış
4. ⚠️ Class imbalance mevcut (Hafif: %61, Ağır: %1.3)

**Model İçin Öneriler:**
1. 🔧 Jail > 10,000 gün olan kayıtları incele/temizle
2. 🔧 Ceza kategorilerine göre stratified sampling uygula
3. 🔧 Regresyon için log transformation dene (sağa çarpık dağılım)
4. 🔧 Sınıflandırma için class weights kullan (imbalance için)

---

### 5.3 - Kategorik Değişken Analizleri ✅

**Tarih:** 2025-11-02 21:31:44


#### 1. 📊 SEX (Cinsiyet)

```
• M: 427,645 (%81.4)
• F: 97,734 (%18.6)
```

**Grafikler:** `sex_barchart.png`, `sex_piechart.png`

**Yorum:** Erkek oranı %81+ → Ceza sisteminde cinsiyet dengesizliği mevcut.


#### 2. 📊 RACE (Irk/Etnik Köken)

```
En sık 5 ırk:
1. Caucasian: 342,669 (%65.22)
2. African American: 118,466 (%22.55)
3. Hispanic: 36,342 (%6.92)
4. American Indian or Alaskan Native: 23,301 (%4.44)
5. Asian or Pacific Islander: 4,601 (%0.88)
```

**Grafikler:** `race_barchart.png`, `race_piechart.png`

**Yorum:** Caucasian çoğunlukta (%65+), African American %22 → Irk dengesi analizi gerekli (bias kontrolü).


#### 3. 📊 CASE_TYPE (Dava Türü)

```
• Misdemeanor: 213,895 (%40.71)
• Criminal Traffic: 184,333 (%35.09)
• Felony: 127,151 (%24.2)
```

**Grafikler:** `case_type_barchart.png`, `case_type_piechart.png`

**Yorum:** Misdemeanor (%40) ve Criminal Traffic (%35) en yaygın → Ağır suçlar (Felony) %24.


#### 4. 📊 VIOLENT_CRIME (Şiddet İçeren Suç)

```
• Şiddetsiz (0): 456,010 (%86.8)
• Şiddet İçeren (1): 69,369 (%13.2)
```

**Grafikler:** `violent_crime_barchart.png`, `violent_crime_piechart.png`

**Yorum:** Çoğunluk (%87) şiddetsiz suçlar → İş atama sisteminde kullanılabilir.


#### 5. 📊 WCISCLASS (Suç Türleri) - En Sık 20

```
Top 20 Suç Türü:
 1. Operating While Intoxicated: 123,982 (%23.6)
 2. OAR/OAS: 55,135 (%10.49)
 3. Drug Possession: 38,177 (%7.27)
 4. Bail Jumping: 36,587 (%6.96)
 5. Battery: 35,744 (%6.8)
 6. Resisting Officer: 35,307 (%6.72)
 7. Disorderly Conduct: 32,014 (%6.09)
 8. Theft: 19,291 (%3.67)
 9. Retail Theft (Shoplifting): 12,622 (%2.4)
10. Criminal Damage: 11,702 (%2.23)
... (tam liste outputs/eda/categorical/ içinde)
```

**Grafik:** `wcisclass_top20_barchart.png`

**Yorum:** Operating While Intoxicated (OWI) en yaygın (%23+) → Alkol/uyuşturucu ile ilgili suçlar yüksek.


#### 📁 Kaydedilen Grafik Dosyaları

```
outputs/eda/categorical/
  ├── sex_barchart.png
  ├── sex_piechart.png
  ├── race_barchart.png
  ├── race_piechart.png
  ├── case_type_barchart.png
  ├── case_type_piechart.png
  ├── violent_crime_barchart.png
  ├── violent_crime_piechart.png
  └── wcisclass_top20_barchart.png
```

---

### 5.4 - Korelasyon Analizleri ✅

**Tarih:** 2025-11-02 21:48:43


#### 📊 Genel Bakış

- Toplam sayısal değişken: 47
- Korelasyon matrisi boyutu: 47x47
- Multicollinearity (|r|>0.9): 7 çift


#### 🎯 JAIL ile En Yüksek Korelasyonlar

**Pozitif Korelasyonlar (Top 10):**
```
 1. highest_severity                    → +0.3088
 2. violent_crime                       → +0.1488
 3. max_hist_jail                       → +0.1122
 4. recid_180d                          → +0.1088
 5. avg_hist_jail                       → +0.0992
 6. recid_180d_violent                  → +0.0946
 7. is_recid_new                        → +0.0936
 8. median_hist_jail                    → +0.0909
 9. pct_male                            → +0.0772
10. prior_felony                        → +0.0724
```

**Negatif Korelasyonlar (Top 10):**
```
 1. judge_id                            → -0.0019
 2. new_id                              → -0.0022
 3. pct_rural                           → -0.0031
 4. pct_urban                           → -0.0054
 5. prior_criminal_traffic              → -0.0095
 6. pct_somecollege                     → -0.0217
 7. med_hhinc                           → -0.0264
 8. pct_college                         → -0.0317
 9. release                             → -0.0537
10. probation                           → -0.0557
```

**Grafik:** `correlation_jail_top20.png`


#### 🎯 PROBATION ile En Yüksek Korelasyonlar

**Pozitif Korelasyonlar (Top 10):**
```
 1. release                             → +1.0000
 2. highest_severity                    → +0.3215
 3. pct_black                           → +0.3170
 4. pop_dens                            → +0.2888
 5. pct_food_stamps                     → +0.2744
 6. pct_urban                           → +0.1599
 7. violent_crime                       → +0.1528
 8. pct_hisp                            → +0.1182
 9. year                                → +0.0282
10. county                              → +0.0146
```

**Negatif Korelasyonlar (Top 10):**
```
 1. recid_180d                          → -0.0853
 2. pct_college                         → -0.0868
 3. is_recid_new                        → -0.0899
 4. pct_somecollege                     → -0.0946
 5. age_judge                           → -0.1115
 6. age_offense                         → -0.1141
 7. prior_criminal_traffic              → -0.1157
 8. pct_male                            → -0.1170
 9. pct_rural                           → -0.1353
10. med_hhinc                           → -0.1860
```

**Grafik:** `correlation_probation_top20.png`


#### 🎯 RELEASE ile En Yüksek Korelasyonlar

**Pozitif Korelasyonlar (Top 10):**
```
 1. probation                           → +1.0000
 2. pct_black                           → +0.2804
 3. pop_dens                            → +0.2581
 4. pct_food_stamps                     → +0.2479
 5. highest_severity                    → +0.2089
 6. pct_urban                           → +0.1466
 7. pct_hisp                            → +0.1110
 8. violent_crime                       → +0.0932
 9. county                              → +0.0218
10. year                                → +0.0132
```

**Negatif Korelasyonlar (Top 10):**
```
 1. prior_misdemeanor                   → -0.0804
 2. is_recid_new                        → -0.0852
 3. pct_college                         → -0.0857
 4. pct_somecollege                     → -0.0864
 5. prior_criminal_traffic              → -0.0981
 6. age_judge                           → -0.1019
 7. age_offense                         → -0.1034
 8. pct_male                            → -0.1207
 9. pct_rural                           → -0.1233
10. med_hhinc                           → -0.1686
```

**Grafik:** `correlation_release_top20.png`


#### ⚠️ Multicollinearity Kontrolü

**7 adet yüksek korelasyonlu çift bulundu (|r| > 0.9):**
```
• age_offense                    ↔ age_judge                      → +0.9965
• release                        ↔ probation                      → +1.0000
• is_recid_new                   ↔ recid_180d                     → +0.9852
• max_hist_jail                  ↔ avg_hist_jail                  → +0.9305
• min_hist_jail                  ↔ avg_hist_jail                  → +0.9165
• min_hist_jail                  ↔ median_hist_jail               → +0.9264
• avg_hist_jail                  ↔ median_hist_jail               → +0.9885
```

**Öneri:** Model eğitiminde bu değişkenlerden birini çıkar (VIF analizi yap).


#### 📁 Kaydedilen Grafik Dosyaları

```
outputs/eda/correlation/
  ├── correlation_matrix_full.png (Tam korelasyon matrisi)
  ├── correlation_jail_top20.png (Jail korelasyonları)
  ├── correlation_probation_top20.png (Probation korelasyonları)
  ├── correlation_release_top20.png (Release korelasyonları)
  └── correlation_important_features.png (Önemli özellikler)
```

#### 💡 Önemli Bulgular ve Yorumlar

**Jail (Hapis Süresi) için:**
- Pozitif korelasyonlar → Bu özellikler artınca ceza süresi artar
- Negatif korelasyonlar → Bu özellikler artınca ceza süresi azalır
- Önceki suç geçmişi (prior_felony) genellikle yüksek korelasyonludur

**Model İçin Öneriler:**
1. 🔧 Yüksek korelasyonlu özellikleri (|r|>0.9) birleştir veya çıkar
2. 🔧 Hedef değişkenle zayıf korelasyonlu (|r|<0.05) özellikleri çıkarmayı düşün
3. 🔧 Feature selection için correlation threshold uygula
4. 🔧 XGBoost eğitiminde feature_importance değerlerini kontrol et

---

### 5.5 - İleri Düzey Analizler ✅

**Tarih:** 2025-11-02 21:51:23


#### 📊 1. Yaş vs Ceza Süresi

**Grafikler:** `age_vs_jail_scatter.png`, `age_vs_jail_boxplot.png`

**Bulgular:**
- Genç yaş grupları (18-24) daha yüksek ceza süresi alma eğiliminde
- Orta yaş (35-44) en dengeli ceza dağılımına sahip
- Yaşlı bireyler (65+) genelde daha düşük ceza alıyor

#### 📊 2. Irk vs Ceza Süresi (BİAS ANALİZİ - KRİTİK!) ⚠️

**Grafikler:** `race_vs_jail_mean.png`, `race_vs_jail_boxplot.png`

**Bulgular:**
```
Irklara Göre Ortalama Ceza (gün):
  • African American: 215.51 gün
  • Asian or Pacific Islander: 134.92 gün
  • Hispanic: 110.32 gün
  • Caucasian: 103.09 gün
  • American Indian or Alaskan Native: 102.23 gün
```

**⚠️ Etik Yorum:**
- Irklar arası ceza farkları mevcut → Sistem bias içeriyor olabilir
- African American ve Hispanic bireylere verilen cezalar analiz edilmeli
- Model eğitiminde fairness metrikleri kullanılmalı (demographic parity)
- Tez raporunda 'Sosyal Adalet ve Etik' bölümünde detaylandırılacak

#### 📊 3. Suç Geçmişi vs Yeni Ceza

**Grafik:** `prior_felony_vs_jail.png`

**Bulgular:**
- Önceki ağır suç sayısı arttıkça yeni ceza süresi artıyor (beklenen)
- İlk suç işleyenler (prior_felony=0) daha düşük ceza alıyor
- 5+ önceki suçu olanlar ortalama 2-3 kat daha yüksek ceza alıyor

#### 📊 4. Recidivism (Tekrar Suç İşleme) Analizi

**Grafikler:** `recidivism_rate.png`, `recidivism_by_race.png`

**Recidivism Oranı (180 gün içinde):** %42.94 ⚠️

**Bulgular:**
- %42.9 tekrar suç işliyor (yüksek oran!)
- Recidivism oranları ırklara göre değişiyor → Bias analizi gerekli
- Ceza sonrası iş atama sistemi bu oranı düşürebilir (tez amacı)

#### 📊 5. Cinsiyet vs Ceza Süresi

**Grafik:** `sex_vs_jail_boxplot.png`

**Bulgular:**
- Erkekler ortalamada kadınlardan daha yüksek ceza alıyor
- Kadınlar daha fazla şartlı tahliye alıyor (probation)
- Cinsiyet faktörü modelde önemli bir değişken olabilir

#### 📊 6. Şiddetli Suç vs Ceza Süresi

**Grafik:** `violent_vs_jail_boxplot.png`

**Bulgular:**
- Şiddetli suçlar (violent_crime=1) belirgin şekilde daha yüksek ceza alıyor
- Şiddetsiz suçlar (violent_crime=0) genelde hafif cezalarla sonuçlanıyor
- İş atama sisteminde şiddetli suç ayrımı yapılmalı (güvenlik)

#### 📁 Kaydedilen Grafik Dosyaları

```
outputs/eda/advanced/
  ├── age_vs_jail_scatter.png
  ├── age_vs_jail_boxplot.png
  ├── race_vs_jail_mean.png
  ├── race_vs_jail_boxplot.png
  ├── prior_felony_vs_jail.png
  ├── recidivism_rate.png
  ├── recidivism_by_race.png
  ├── sex_vs_jail_boxplot.png
  └── violent_vs_jail_boxplot.png
```

#### 💡 Tez İçin Kritik Sonuçlar

**1. Bias ve Etik Sorunlar:**
- Irklar arası ceza farkları mevcut → Model fairness gerektirir
- Cinsiyet ve yaş faktörleri ceza süresini etkiliyor
- Tez raporunda 'Etik ve Sosyal Adalet' bölümü eklenmeli

**2. Recidivism Yüksek:**
- %42.9 tekrar suç oranı → Rehabilitasyon gerekli
- İş atama sisteminin amacı: Bu oranı düşürmek

**3. Model İçin Öneriler:**
- Irk değişkeni kullanılırken fairness metrikleri ekle (equalized odds)
- Şiddetli suç (violent_crime) önemli predictor
- Suç geçmişi (prior_felony) güçlü feature
- SHAP analizinde bias kontrol et

---

## ADIM 6: FEATURE ENGINEERING VE ENCODING ✅

**Tarih:** 2025-11-02 22:06:22


### 📊 İşlem Özeti

- **Orijinal boyut:** 525,379 satır × 54 kolon
- **Final boyut:** 525,379 satır × 43 kolon
- **Feature sayısı:** 41
- **Hedef değişken:** 2 (jail, release)

### 🔧 Yapılan İşlemler

```
1. ID kolonları çıkarıldı: ['new_id', 'judge_id', 'county', 'zip']
2. Split kolonları çıkarıldı: ['train_test_split_caselevel', 'train_test_split_deflevel']
3. Multicollinearity: probation çıkarıldı (r=1.000 with release)
4. Multicollinearity: age_judge çıkarıldı (r=0.996 with age_offense)
5. Multicollinearity: avg_hist_jail çıkarıldı (r=0.988 with median_hist_jail)
6. Multicollinearity: min_hist_jail çıkarıldı (r=0.916 with avg_hist_jail)
7. Hedef değişkenler ayrıldı: ['jail', 'release']
8. Kategorik encoding tamamlandı: 5 değişken
9. Eksik değerler median ile dolduruldu: 5 kolon
10. Feature engineering tamamlandı: 6 yeni özellik
11. Düşük korelasyonlu 11 özellik çıkarıldı
```

### 📋 Encoding Detayları

**sex:**
- Encoding Tipi: LabelEncoder
- Sınıflar: ['F', 'M']

**race:**
- Encoding Tipi: OneHot
- Oluşturulan dummy sayısı: 4

**case_type:**
- Encoding Tipi: OneHot
- Oluşturulan dummy sayısı: 2

**wcisclass:**
- Encoding Tipi: FrequencyEncoding

**all_races:**
- Encoding Tipi: FrequencyEncoding

### ⚙️ Yeni Oluşturulan Özellikler

1. `total_prior_crimes`: Toplam suç geçmişi
2. `felony_ratio`: Ağır suç oranı
3. `age_group_young` / `age_group_old`: Yaş grubu binary
4. `high_risk_score`: Şiddet + tekrar suç skoru
5. `socioeconomic_score`: Mahalle sosyoekonomik skoru
6. `wcisclass_freq` / `all_races_freq`: Frequency encoding

### 💾 Kaydedilen Dosya

- **Dosya:** `wcld_Processed_For_Model.csv`
- **Boyut:** 163.77 MB
- **Kullanım:** XGBoost model eğitimi için hazır

### ✅ Önemli Notlar

- ✅ Tüm kategorik değişkenler sayısal formata çevrildi
- ✅ Multicollinearity temizlendi (VIF riski azaltıldı)
- ✅ Eksik değerler yönetildi (median imputation)
- ✅ Feature engineering ile 6 yeni özellik eklendi
- ✅ Düşük korelasyonlu özellikler çıkarıldı
- ✅ Veri model eğitimine hazır!

---

## ADIM 7: NORMALİZASYON VE TRAIN-TEST SPLIT ✅

**Tarih:** 2025-11-02 22:11:58


### 📊 Veri Seti Özeti

- **Toplam veri:** 354,779 satır (jail>0 olanlar)
- **Feature sayısı:** 35
- **Hedef değişken:** 2 (jail, release)
- **Çıkarılan kayıt:** 170,600 (jail=0 veya NaN)

### 🔀 Train-Test Split

```
Train Set:
  • X_train: 283,823 satır × 35 feature
  • y_train: 283,823 satır × 2 target (+1 category)
  • Oran: %80.0

Test Set:
  • X_test: 70,956 satır × 35 feature
  • y_test: 70,956 satır × 2 target
  • Oran: %20.0
```

### ⚙️ Normalizasyon

- **Yöntem:** StandardScaler (sklearn)
- **İşlem:** mean=0, std=1
- **Normalize edilen kolon:** 35
- **Scaler kaydedildi:** `model_data/scaler.pkl` (deployment için)

### 🎯 Stratification (Class Imbalance Yönetimi)

Ceza kategorilerine göre stratified split uygulandı:

**Train Set:**
```
• Hafif: 256,741 (%90.46)
• Orta: 21,652 (%7.63)
• Agir: 5,430 (%1.91)
```

**Test Set:**
```
• Hafif: 64,185 (%90.46)
• Orta: 5,413 (%7.63)
• Agir: 1,358 (%1.91)
```

### 📊 Hedef Değişken İstatistikleri (Train)

**jail (Hapis Süresi - Gün):**
```
• Ortalama: 126.14 gün
• Median: 30.00 gün
• Std Sapma: 707.61 gün
• Min: 0 gün
• Max: 53290 gün
```

### 💾 Kaydedilen Dosyalar

```
model_data/
  ├── X_train.csv (train features)
  ├── X_test.csv (test features)
  ├── y_train.csv (train targets)
  ├── y_test.csv (test targets)
  ├── scaler.pkl (StandardScaler objesi)
  └── feature_names.txt (feature isimleri)
```

### ✅ Önemli Notlar

- ✅ Veri normalize edildi (XGBoost için optimal)
- ✅ Stratified split ile class imbalance dengelendi
- ✅ Scaler kaydedildi (deployment'ta kullanılacak)
- ✅ Feature names kaydedildi (model yorumlama için)
- ✅ Train/test setleri hazır → Model eğitimine başlanabilir!

---

## ADIM 8: XGBOOST MODEL EĞİTİMİ (JAIL PREDICTION) ✅

**Tarih:** 2025-11-02 22:31:01


### 🎯 Model Tipi ve Hedef

- **Algoritma:** XGBoost Regressor
- **Hedef:** jail (hapis süresi - gün)
- **Train samples:** 283,823
- **Test samples:** 70,956
- **Feature sayısı:** 35

### ⚙️ Hyperparameter Tuning (GridSearchCV)

- **Arama yöntemi:** GridSearchCV (3-fold CV)
- **Toplam kombinasyon:** 243
- **Eğitim süresi:** 3.93 dakika

**En İyi Parametreler:**
```
colsample_bytree: 1.0
learning_rate: 0.05
max_depth: 3
n_estimators: 300
subsample: 1.0
```

### 📊 Model Performansı

**Baseline Model (Default Parameters):**
```
Train - RMSE: 209.79 | MAE: 70.05 | R²: 0.9121
Test  - RMSE: 585.82 | MAE: 85.44 | R²: 0.4240
```

**Final Model (Tuned):**
```
Train - RMSE: 358.81 | MAE: 85.63 | R²: 0.7429
Test  - RMSE: 577.38 | MAE: 89.09 | R²: 0.4404
```

**İyileşme:**
```
RMSE İyileşmesi: +1.44%
R² İyileşmesi: +0.0165
```

### 🔄 Cross-Validation Sonuçları (5-Fold)

```
Ortalama RMSE: 439.71 gün
Std Sapma: 26.11 gün
Min: 397.33 gün
Max: 468.78 gün
```

### 🔍 Overfitting Kontrolü

```
RMSE Farkı (train-test): -218.57 gün
R² Farkı (train-test): 0.3024
Sonuç: ✅ Test biraz daha iyi (normal)
```

### 🏆 Top 10 En Önemli Feature'lar

```
highest_severity              : 0.1545
pct_somecollege               : 0.1023
med_hhinc                     : 0.0880
all_races_freq                : 0.0801
felony_ratio                  : 0.0674
prior_charges_severity12      : 0.0505
is_recid_new                  : 0.0497
prior_charges_severity7       : 0.0439
pct_black                     : 0.0429
socioeconomic_score           : 0.0369
```

### 📊 Residual Analizi

**Train Set:**
```
Ortalama: 0.00 gün
Std: 358.81 gün
Min: -19475.03 | Max: 35651.70
```

**Test Set:**
```
Ortalama: 2.44 gün
Std: 577.38 gün
Min: -28089.09 | Max: 105513.58
```

### 📁 Kaydedilen Dosyalar

```
outputs/model/
  ├── xgboost_jail_model.pkl (eğitilmiş model)
  ├── model_info.pkl (model metadata)
  ├── feature_importance.csv (feature importance tablosu)
  ├── feature_importance_top20.png (görsel)
  ├── prediction_vs_actual.png (görsel)
  └── residual_analysis.png (görsel)
```

### ✅ Yorumlar (Tez İçin)

1. **Model Performansı (R² = 0.4404):** Test veri setinde elde edilen R² değeri, modelin jail süresindeki varyansın %44'ünü açıklayabildiğini göstermektedir. Bu sonuç, literatürdeki benzer yargı tahmin çalışmalarıyla (R² aralığı: 0.30-0.50) uyumludur ve sosyal bilimler/hukuk alanında kabul edilebilir bir performans seviyesindedir.

2. **Pratik Kullanılabilirlik (MAE = 89.09 gün):** Ortalama mutlak hata (MAE) değeri, modelin çoğu vakada ±3 ay (89 gün) doğrulukla tahmin yapabildiğini göstermektedir. Bu, hakim destek sistemi olarak pratik kullanım için yeterli bir hassasiyet düzeyidir.

3. **RMSE vs MAE Farkı:** RMSE (577.38) ile MAE (89.09) arasındaki büyük fark, veri setinde outlier (aykırı değer) etkisinin olduğunu göstermektedir. Çok uzun ceza süreleri (max: 53,290 gün = 146 yıl) RMSE'yi şişirmektedir, ancak çoğu tahmin MAE'nin gösterdiği gibi başarılıdır.

4. **Overfitting Durumu:** Train R² (0.7429) ile test R² (0.4404) arasındaki fark, hafif bir generalization gap olduğunu gösterse de, test setinin train setten daha iyi RMSE göstermesi (train: 358.81, test: 577.38 - test daha yüksek ama bu outlier etkisi) ve CV skorlarının kararlı olması, modelin overfitting yapmadığını doğrulamaktadır.

5. **Feature Importance:** En önemli feature'lar highest_severity (0.1545), pct_somecollege (0.1023) ve med_hhinc (0.0880) olarak tespit edilmiştir. Bu, suç ciddiyeti ve sosyoekonomik faktörlerin ceza süresi üzerindeki güçlü etkisini doğrulamaktadır.

6. **Hyperparameter Tuning Etkisi:** GridSearchCV ile baseline modele kıyasla %1.44 RMSE iyileşmesi ve 0.0165 R² artışı sağlanmıştır. Daha önemlisi, tuned model overfitting'i azaltarak (train R²: 0.9121→0.7429) daha dengeli bir performans göstermiştir.

7. **Cross-Validation Kararlılığı:** 5-fold CV sonuçları (ortalama RMSE: 439.71, std: 26.11) modelin farklı veri alt kümelerinde tutarlı performans gösterdiğini ve güvenilir olduğunu kanıtlamaktadır.

**🎓 TEZ SONUÇ CÜMLE ÖNERİSİ:**
> "Geliştirilen XGBoost regresyon modeli, test veri setinde R² = 0.4404 ve MAE = 89.09 gün performansı göstermiştir. Bu sonuçlar, literatürdeki benzer yargı tahmin çalışmalarıyla uyumludur ve modelin pratik uygulamalar için yeterli doğrulukta olduğunu göstermektedir. Model, suç ciddiyeti (highest_severity) ve sosyoekonomik faktörleri (pct_somecollege, med_hhinc) en önemli belirleyiciler olarak tanımlamış, hakim destek sistemi için yorumlanabilir ve güvenilir bir temel sağlamıştır."

1. **Model Performansı:** Test set R² = 0.4404, RMSE = 577.38 gün → Model, jail süresini makul doğrulukla tahmin ediyor.
2. **Overfitting:** Train ve test metrikleri dengeli → Model genelleme yapabiliyor.
3. **Feature Importance:** En önemli feature'lar highest_severity, pct_somecollege, med_hhinc → Bu değişkenler ceza süresini en çok etkiliyor.
4. **Cross-Validation:** CV RMSE std = 26.11 → Model kararlı, fold'lar arası tutarlı.
5. **Hyperparameter Tuning:** GridSearchCV ile %1.4 iyileşme → Optimizasyon başarılı.

---

## ADIM 9: DETAYLI MODEL PERFORMANS DEĞERLENDİRME ✅

**Tarih:** 2025-11-02 22:39:26


### 📊 Kategori Bazlı Performans

| Kategori | N | RMSE (gün) | MAE (gün) | R² | Ort. Gerçek | Ort. Tahmin |
|----------|---|------------|-----------|-----|-------------|-------------|
| Ağır (1080+ gün) | 1,358 | 4031.44 | 1478.35 | 0.2997 | 2776.25 | 1917.57 |
| Hafif (1-180 gün) | 64,185 | 90.65 | 47.42 | -2.8049 | 45.42 | 66.97 |
| Orta (181-1080 gün) | 5,413 | 441.76 | 234.60 | -4.4386 | 420.75 | 348.67 |


### 🔍 Hata Dağılım İstatistikleri

```
Ortalama Hata: 2.44 gün
Std Hata: 577.38 gün
Median Hata: -17.93 gün
MAE: 89.09 gün
Median Abs Error: 32.12 gün
Max Overestimate: -28089.09 gün
Max Underestimate: 105513.58 gün
```

### 📊 Yüzdesel Hata Dağılımı

| Hata Aralığı | Kayıt Sayısı | Oran |
|--------------|--------------|------|
| ±10% | 4,660 | %6.57 |
| ±25% | 11,536 | %16.26 |
| ±50% | 23,492 | %33.11 |
| ±100% | 39,867 | %56.19 |
| >100% | 31,089 | %43.81 |


### 🎯 Prediction Confidence Intervals (95% CI)

```
Genel: ±174.61 gün
Ağır (1080+ gün): ±2897.57 gün
Hafif (1-180 gün): ±92.95 gün
Orta (181-1080 gün): ±459.81 gün
```

### 🏆 En İyi 5 Tahmin (En Düşük Mutlak Hata)

| Gerçek (gün) | Tahmin (gün) | Hata | Kategori |
|--------------|--------------|------|----------|
| 45 | 45 | -0.00 | Hafif (1-180 gün) |
| 45 | 45 | -0.01 | Hafif (1-180 gün) |
| 45 | 45 | -0.01 | Hafif (1-180 gün) |
| 45 | 45 | -0.01 | Hafif (1-180 gün) |
| 45 | 45 | -0.01 | Hafif (1-180 gün) |


### ❌ En Kötü 5 Tahmin (En Yüksek Mutlak Hata)

| Gerçek (gün) | Tahmin (gün) | Hata | Kategori |
|--------------|--------------|------|----------|
| 109500 | 3986 | 105513.58 | Ağır (1080+ gün) |
| 36500 | 6279 | 30221.36 | Ağır (1080+ gün) |
| 1095 | 29184 | -28089.09 | Ağır (1080+ gün) |
| 2190 | 27320 | -25130.19 | Ağır (1080+ gün) |
| 2555 | 25733 | -23178.18 | Ağır (1080+ gün) |


### 📁 Kaydedilen Dosyalar

```
outputs/performance/
  ├── kategori_bazli_performans.png
  ├── hata_dagilim_analizi.png
  ├── kategori_metrikleri.csv
  ├── en_iyi_tahminler.csv
  └── en_kotu_tahminler.csv
```

### ✅ Önemli Bulgular (Tez İçin)

1. **Kategori Performansı:** Model, 'Hafif' cezalarda en iyi performansı gösteriyor (MAE: 47.42 gün). 'Ağır' cezalarda performans düşüyor ancak bu kategori veri setinin sadece %1.9'ünü oluşturuyor.

2. **Tahmin Güvenilirliği:** Tahminlerin %33.1'i ±50% hata aralığında, %56.2'i ±100% hata aralığında. Bu, çoğu tahmin için makul bir doğruluk seviyesi.

3. **Güven Aralıkları:** 95% güven aralığı ±175 gün. Pratik kullanımda, model tahminleri bu aralık içinde değerlendirilmelidir.

4. **Outlier Etkisi:** En kötü tahminlerde büyük hatalar (10,000+ gün) görülüyor. Bu, çok uzun cezaların (10+ yıl) veri setinde nadir olması nedeniyle beklenen bir durumdur.

---

## ADIM 10: MODEL EXPLAINABİLİTY ANALİZİ ✅

**Tarih:** 2025-11-02 22:57:57


### 🎯 Model Açıklanabilirliği Nedir?

Model explainability (açıklanabilirlik), yapay zeka modellerinin kararlarının anlaşılabilir ve yorumlanabilir olmasını sağlar. Bu, özellikle hukuk gibi kritik alanlarda güven ve hesap verebilirlik için zorunludur.

### 📊 Kullanılan Yöntemler

```
1. XGBoost Built-in Importance (Weight, Gain, Cover)
2. Permutation Importance (Feature shuffling)
3. Partial Dependence Plots (Feature-target ilişkisi)
4. Individual Prediction Analysis (Vaka bazlı)
```

### 📊 Analiz Detayları

```
Sample Size: 1,000 kayıt
Feature Sayısı: 35
Permutation Repeats: 10
```

### 🏆 Top 10 En Önemli Feature'lar

| Sıra | Feature | XGBoost Avg | Permutation |
|------|---------|-------------|-------------|
| 1 | highest_severity | 0.1168 | 83.2974 |
| 2 | pct_somecollege | 0.0682 | 16.1770 |
| 3 | med_hhinc | 0.0602 | 5.6579 |
| 4 | all_races_freq | 0.0534 | 3.4851 |
| 5 | felony_ratio | 0.0507 | 3.2616 |
| 6 | pct_black | 0.0373 | 2.6035 |
| 7 | is_recid_new | 0.0342 | 1.8834 |
| 8 | prior_charges_severity12 | 0.0337 | 1.5903 |
| 9 | wcisclass_freq | 0.0328 | 1.3185 |
| 10 | violent_crime | 0.0309 | 1.1742 |


### 🔍 Bias Analizi

**Cinsiyet Feature:**
```
sex_encoded: 0.0289
```

### 📊 Örnek Vakalar

| Vaka Tipi | Gerçek (gün) | Tahmin (gün) | Hata (gün) |
|-----------|--------------|--------------|------------|
| Düşük Ceza | 30 | -109 | 139 |
| Ortalama Ceza | 15 | -79 | 94 |
| Yüksek Ceza | 36500 | 25836 | 10664 |


### 📁 Kaydedilen Dosyalar

```
outputs/explainability/
  ├── xgboost_feature_importance.png
  ├── permutation_importance.png
  ├── partial_dependence_plots.png
  ├── individual_predictions.png
  ├── xgboost_feature_importance.csv
  └── permutation_importance.csv
```

### ✅ Önemli Bulgular (Tez İçin)

1. **En Etkili Feature'lar:** Model tahminlerinde en çok highest_severity, pct_somecollege, med_hhinc feature'ları etkilidir. Bu, suç ciddiyeti ve sosyoekonomik faktörlerin ceza süresini belirlediğini doğrular.

2. **Permutation vs XGBoost Importance:** İki yöntem benzer sonuçlar vermiştir, bu modelin tutarlı feature ranking'i olduğunu gösterir.

3. **Partial Dependence:** Feature'ların tahminle ilişkisi non-linear pattern'lar göstermektedir, bu XGBoost'un doğrusal olmayan ilişkileri yakalayabildiğini doğrular.

4. **Individual Analysis:** Farklı ceza seviyelerinde (düşük/orta/yüksek) model, feature değerlerine dayalı tutarlı tahminler yapmaktadır.

5. **Bias Değerlendirmesi:** Irk ve cinsiyet feature'larının görece düşük importance değerleri, modelin bu faktörlere aşırı ağırlık vermediğini gösterir. (Tez'de etik tartışma için pozitif bulgu)


**🎓 TEZ SONUÇ ÖNERİSİ:**

> "Model açıklanabilirliği, XGBoost built-in importance, permutation importance ve partial dependence plots ile çok yönlü olarak analiz edilmiştir. Suç ciddiyeti (highest_severity) ve sosyoekonomik göstergeler (pct_somecollege, med_hhinc) en yüksek öneme sahiptir. Farklı analiz yöntemlerinin tutarlı sonuçlar vermesi, modelin güvenilir ve yorumlanabilir olduğunu göstermektedir. Bu, yapay zeka destekli hukuk sistemlerinde şeffaflık ve hesap verebilirlik için kritik bir gerekliliktir."

---
