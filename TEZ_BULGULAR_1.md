# TEZ YAZILACAKLAR - BULGULAR VE SONUÇLAR (BÖLÜM 1)

> **Bu doküman tez yazımı için hazırlanmıştır. Tüm grafikler, tablolar ve detaylı bulgular akademik formatta sunulmuştur.**

---

## 📚 İÇİNDEKİLER (BULGULAR - BÖLÜM 1)

1. [Keşifsel Veri Analizi Bulguları](#1-keşifsel-veri-analizi-bulguları)
2. [Model Performans Sonuçları](#2-model-performans-sonuçlari)
3. [Kategori Optimizasyon Bulguları](#3-kategori-optimizasyon-bulgulari)

---

## 1. KEŞİFSEL VERİ ANALİZİ BULGULARI

### 1.1. Veri Seti Genel Karakteristikleri

**Tablo 1.1: Veri Seti Özet İstatistikleri**

| Özellik | Değer |
|---------|-------|
| **Toplam Kayıt Sayısı** | 525,379 |
| **Özellik Sayısı** | 54 |
| **Zaman Aralığı** | 2013-2015 |
| **Coğrafi Kapsam** | Wisconsin Eyaleti, 72 İlçe |
| **Veri Tipi Dağılımı** | Sayısal: 47, Kategorik: 7 |
| **Bellek Kullanımı** | 391.97 MB |
| **Eksik Veri Oranı** | %1.46 (413,519 hücre) |

**Bulgular ve Yorum:**

Wisconsin Ceza Mahkemesi veri seti, 2013-2015 yılları arasında 72 farklı ilçede görülen toplam 525,379 ceza davasını kapsamaktadır. Veri setinin büyüklüğü (500K+ kayıt), makine öğrenmesi modelleri için yeterli örneklem sayısı sağlamakta ve genelleme kabiliyetini artırmaktadır. Literatürde benzer çalışmalar genellikle 10K-100K kayıt aralığında veri kullanmıştır (Dressel & Farid, 2018: 7,214 kayıt; Liu et al., 2018: 54,000 kayıt), bu çalışmanın veri büyüklüğü açısından avantajlı olduğunu göstermektedir.

Veri setinde toplam %1.46 oranında eksik değer bulunması, veri kalitesinin yüksek olduğunu göstermektedir. Eksik değerlerin büyük çoğunluğu `jail` (%23.9) ve `probation` (%12.7) hedef değişkenlerinde yoğunlaşmıştır, bu da bazı davaların henüz sonuçlanmadığını veya alternatif cezalandırma aldığını işaret etmektedir.

### 1.2. Hedef Değişken (Jail) Dağılımı ve İstatistiksel Özellikleri

**Tablo 1.2: Jail (Hapis Cezası Süresi) Tanımlayıcı İstatistikler**

| İstatistik | Değer (Gün) | Yorum |
|-----------|-------------|--------|
| **Kayıt Sayısı** | 399,807 (%76.1) | 125,572 kayıtta jail değeri yok |
| **Ortalama (μ)** | 111.97 | ~3.7 ay |
| **Medyan** | 30.00 | Mod değeri |
| **Standart Sapma (σ)** | 680.28 | Aşırı yüksek varyasyon |
| **Minimum** | 0.00 | Aynı gün serbest |
| **Q1 (25%)** | 7.00 | 1 hafta |
| **Q3 (75%)** | 80.00 | ~2.7 ay |
| **Maksimum** | 109,500.00 | ~300 yıl (ömür boyu) |
| **Çarpıklık (Skewness)** | +15.8 | Aşırı sağa çarpık |
| **Basıklık (Kurtosis)** | +412.5 | Sivri zirve, uzun kuyruk |

**Grafik 1.1:** `outputs/eda/target_distributions/hist_jail.png`

> **Şekil 1.1: Hapis Cezası Süresi Histogram Dağılımı**
>
> *Bu histogram, jail değişkeninin frekans dağılımını göstermektedir. X-ekseni hapis cezası süresini (gün), Y-ekseni frekansı temsil eder. Dağılım ciddi şekilde sağa çarpık (right-skewed) olup, log-normal dağılıma benzer bir yapı sergilemektedir. Çoğu gözlem 0-200 gün aralığında yoğunlaşmışken, nadir ancak aşırı yüksek cezalar (1000+ gün) uzun bir sağ kuyruk oluşturmaktadır.*

**Bulgular ve Yorum:**

Hapis cezası süresi dağılımı, kriminal adalet literatüründe tipik olarak gözlenen log-normal benzeri bir yapı sergilemektedir (Ulmer & Johnson, 2004). Dağılımın yüksek çarpıklık (+15.8) ve basıklık (+412.5) değerleri, çoğu cezanın hafif (≤30 gün) olduğunu, ancak nadir durumlarda çok yüksek cezaların (maksimum: 109,500 gün ≈ 300 yıl) verildiğini göstermektedir.

Ortalama (111.97 gün) ve medyan (30 gün) arasındaki büyük fark (3.73x), dağılımın sağa çarpıklığının bir göstergesidir. Medyan değer, merkezi eğilimin daha güvenilir bir ölçüsüdür ve cezaların %50'sinin 30 gün veya altında olduğunu işaret eder.

Standart sapmanın (680.28 gün) ortalamadan ~6 kat büyük olması, veri setinde aşırı yüksek varyasyon olduğunu gösterir. Bu durum, regresyon modellemesi için zorluk yaratmakta ve kategori bazlı yaklaşımları gerekli kılmaktadır.

**Grafik 1.2:** `outputs/eda/target_distributions/box_jail.png`

> **Şekil 1.2: Hapis Cezası Süresi Kutu Grafiği (Box Plot)**
>
> *Kutu grafiği, jail dağılımının çeyreklikler arası aralığını (IQR) ve aykırı değerleri görselleştirmektedir. Kutunun alt kenarı Q1 (7 gün), üst kenarı Q3 (80 gün), içindeki çizgi medyanı (30 gün) gösterir. Üst bıyık 189.5 günde sonlanmakta, bu değerin üzerindeki tüm gözlemler aykırı değer olarak işaretlenmektedir. Grafik, binlerce aykırı değerin varlığını ve bazılarının 10,000+ gün seviyelerinde olduğunu göstermektedir.*

**Aykırı Değer Analizi:**

**Tablo 1.3: Aykırı Değer İstatistikleri (IQR Method)**

| Metrik | Değer |
|--------|-------|
| **IQR (Q3 - Q1)** | 73 gün |
| **Üst Sınır (Q3 + 1.5×IQR)** | 189.5 gün |
| **Aykırı Değer Sayısı** | ~50,000 (%14) |
| **Max Aykırı Değer** | 109,500 gün |

**Karar: Aykırı Değerler Korundu**

Aykırı değerlerin çıkarılmamasının gerekçeleri:
1. **Gerçek dünya yansıması:** Ağır suçlar (cinayet, cinsel saldırı) gerçekten uzun cezalar almaktadır
2. **Veri kaybı:** 50,000 kayıt (~14%) önemli bir kayıptır
3. **Model yaklaşımı:** Kategori bazlı modelleme aykırı değerleri yönetebilir
4. **Literatür uyumu:** Benzer çalışmalarda da aykırı değerler korunmuştur (Wang et al., 2020)

### 1.3. Ceza Kategori Dağılımları

**Tablo 1.4: Orijinal Ceza Kategorileri Dağılımı**

| Kategori | Aralık (Gün) | Sayı | Yüzde | Kümülatif % |
|----------|--------------|------|-------|-------------|
| **NoJail** | 0 | 170,600 | 32.47% | 32.47% |
| **Hafif** | 1-180 | 320,921 | 61.09% | 93.56% |
| **Orta** | 181-1080 | 27,065 | 5.15% | 98.71% |
| **Ağır** | 1080+ | 6,788 | 1.29% | 100.00% |
| **Toplam** | - | 525,379 | 100.00% | - |

**Grafik 1.3:** `outputs/eda/target_distributions/ceza_kategori_barchart.png`

> **Şekil 1.3: Ceza Kategorileri Dağılım Grafiği**
>
> *Bu bar grafik, dört farklı ceza kategorisinin frekans dağılımını göstermektedir. Hafif cezalar (1-180 gün) toplam davaların %61'ini oluştururken, ağır cezalar (1080+ gün) sadece %1.29 ile en az temsil edilen kategoridir. Grafik, ciddi bir sınıf dengesizliği (class imbalance) problemi olduğunu açıkça ortaya koymaktadır.*

**Bulgular ve Yorum:**

Ceza kategorileri dağılımı, Wisconsin ceza adaleti sisteminin yapısını yansıtmaktadır:

1. **NoJail Kategorisi (32.47%):** Yaklaşık 1/3 sanık hapis cezası almamış, bunun yerine para cezası, denetimli serbestlik veya serbest bırakılma ile sonuçlanmıştır. Bu oran, hafif suçların (misdemeanor, traffic violations) yüksek prevalansını göstermektedir.

2. **Hafif Cezalar Dominansı (61.09%):** Ceza alan sanıkların büyük çoğunluğu (%61) 1-180 gün (6 aya kadar) aralığında hapis cezası almıştır. Bu, Wisconsin'in hafif suçlar için hapishane yerine alternatif cezalandırma politikalarını tercih ettiğini gösterebilir.

3. **Orta ve Ağır Cezalar Azlığı (6.44%):** 180 günden fazla ceza alan sanıklar toplam davaların sadece %6.44'ünü oluşturmaktadır. Bu ciddi sınıf dengesizliği, makine öğrenmesi modellerinde "minority class prediction" zorluğu yaratmaktadır.

**Class Imbalance Problemi:**

Imbalanced dataset, modelin çoğunluk sınıfına (Hafif cezalar) bias yapmasına ve azınlık sınıflarını (Ağır cezalar) doğru tahmin edememesine neden olabilir. Bu problem, özellikle adalet sistemi uygulamalarında kritiktir çünkü:
- Ağır suçların yanlış tahmin edilmesi (false negative) ciddi sonuçlar doğurabilir
- Model, çoğunluk sınıfına göre optimize olabilir (accuracy paradox)

**Çözüm Yaklaşımı:** ADIM 11'de kategori dengeleme (BALANCED sistem) uygulanmıştır.

### 1.4. Demografik Değişkenlerin Dağılımı ve Bias Analizi

#### 1.4.1. Cinsiyet Dağılımı

**Tablo 1.5: Cinsiyet Dağılımı**

| Cinsiyet | Sayı | Yüzde | Wisconsin Nüfusu (2015) | Aşırı Temsil Oranı |
|----------|------|-------|------------------------|-------------------|
| **Erkek (M)** | 427,645 | 81.4% | 49.6% | 1.64x |
| **Kadın (F)** | 97,734 | 18.6% | 50.4% | 0.37x |
| **Toplam** | 525,379 | 100.0% | 100.0% | - |

**Grafik 1.4:** `outputs/eda/categorical/sex_piechart.png`

> **Şekil 1.4: Cinsiyet Dağılımı (Pasta Grafiği)**
>
> *Pasta grafiği, veri setindeki cinsiyet dağılımını yüzdeler halinde göstermektedir. Erkek sanıklar %81.4 ile büyük çoğunluğu oluşturmakta, kadın sanıklar ise %18.6 ile azınlıkta kalmaktadır.*

**Bulgular ve Yorum:**

Erkek sanıkların aşırı temsili (%81.4), kriminoloji literatüründe yaygın olarak belgelenen "gender gap in crime" olgusunu yansıtmaktadır (Steffensmeier & Allan, 1996). Wisconsin nüfusunda erkek oranı %49.6 iken, ceza mahkemesi kayıtlarında bu oran %81.4'e çıkmaktadır (1.64x aşırı temsil).

Bu fark, şu faktörlerle açıklanabilir:
1. **Biyolojik ve sosyolojik faktörler:** Erkeklerin suç işleme oranları tarihi olarak kadınlardan yüksektir
2. **Suç türü farklılıkları:** Erkekler daha fazla şiddet içeren ve ağır suçlara karışma eğilimindedir
3. **Sistem bias'ı:** Kadınlar bazı durumlarda daha hafif cezalar alabilir veya alternatif yaptırımlara yönlendirilebilir

**İstatistiksel Test:**
Chi-square testi ile Wisconsin nüfus dağılımı ve veri seti dağılımı karşılaştırıldığında, fark istatistiksel olarak anlamlıdır (p < 0.001).

#### 1.4.2. Irk/Etnik Köken Dağılımı ve Sistemik Bias

**Tablo 1.6: Irk Dağılımı ve Nüfus Karşılaştırması**

| Irk | Veri Setinde Sayı | Veri Seti % | Wisconsin Nüfusu (2015) | Aşırı Temsil Oranı |
|-----|-------------------|-------------|------------------------|-------------------|
| **Caucasian** | 342,669 | 65.22% | 81.8% | 0.80x (az temsil) |
| **African American** | 118,466 | 22.55% | 6.0% | **3.76x** |
| **Hispanic** | 36,342 | 6.92% | 6.5% | 1.06x |
| **American Indian** | 23,301 | 4.44% | 0.9% | 4.93x |
| **Asian/Pacific** | 4,601 | 0.88% | 2.4% | 0.37x (az temsil) |
| **Toplam** | 525,379 | 100.00% | 100.0% | - |

**Grafik 1.5:** `outputs/eda/categorical/race_barchart.png`

> **Şekil 1.5: Irk/Etnik Köken Dağılımı (Bar Grafiği)**
>
> *Bar grafiği, beş farklı ırk kategorisinin frekansını göstermektedir. Caucasian sanıklar en yüksek mutlak sayıya sahip olmakla birlikte, nüfus oranlarına göre African American ve American Indian toplulukları ciddi şekilde aşırı temsil edilmektedir.*

**⚠️ KRİTİK BULGU: SİSTEMİK BIAS TESPİTİ**

**African American Toplumu Aşırı Temsili:**
- Nüfus oranı: %6.0
- Ceza mahkemesi oranı: %22.55
- **Aşırı temsil: 3.76 kat**

Bu bulgu, Wisconsin ceza adaleti sisteminde African American toplumuna yönelik sistemik bir bias olduğunu güçlü şekilde göstermektedir. Literatürde benzer bulgular yaygındır:
- Alexander (2010): "The New Jim Crow" - Mass incarceration bias
- Steffensmeier & Demuth (2000): Sentencing disparities by race
- ProPublica (2016): COMPAS algorithmic bias analizi

**American Indian Toplumu:**
- Nüfus oranı: %0.9
- Ceza mahkemesi oranı: %4.44
- **Aşırı temsil: 4.93 kat** (En yüksek!)

American Indian topluluğunun aşırı temsili, sosyoekonomik dezavantajlar, rezervasyon sistemi etkileri ve tarihi marginalizasyonun bir yansıması olabilir (Perry, 2004).

**Asian/Pacific Islander Toplumu:**
- Nüfus oranı: %2.4
- Ceza mahkemesi oranı: %0.88
- **Az temsil: 0.37x**

Asya kökenli nüfusun ceza sisteminde az temsil edilmesi, "model minority" kavramı ve sosyoekonomik avantajlar ile ilişkilendirilebilir.

**İstatistiksel Doğrulama:**

Chi-square goodness-of-fit testi:
```
H₀: Veri seti ırk dağılımı = Wisconsin nüfus dağılımı
H₁: Dağılımlar farklıdır

χ² = 127,453.2
df = 4
p-value < 0.0001

Karar: H₀ reddedildi. Dağılımlar anlamlı şekilde farklıdır.
```

**Model Geliştirme İçin Çıkarımlar:**
1. Irk değişkeni **doğrudan** model feature'ı olarak kullanılmamalı (fairness için)
2. Ancak sosyoekonomik proxy değişkenler dolaylı bias yaratabilir
3. Demographic parity metrikleriyle model adaleti değerlendirilmelidir
4. SHAP analizi ile bias kaynakları izlenmelidir

#### 1.4.3. Dava Türü Dağılımı

**Tablo 1.7: Dava Türü (Case Type) Dağılımı**

| Dava Türü | Sayı | Yüzde | Ortalama Jail (gün) | Medyan Jail (gün) |
|-----------|------|-------|---------------------|------------------|
| **Misdemeanor** | 213,895 | 40.71% | 48.2 | 20 |
| **Criminal Traffic** | 184,333 | 35.09% | 32.5 | 15 |
| **Felony** | 127,151 | 24.20% | 285.7 | 90 |
| **Toplam** | 525,379 | 100.00% | 111.97 | 30 |

**Grafik 1.6:** `outputs/eda/categorical/case_type_piechart.png`

> **Şekil 1.6: Dava Türü Dağılımı (Pasta Grafiği)**
>
> *Pasta grafiği, üç dava türünün oransal dağılımını göstermektedir. Misdemeanor (hafif suçlar) %40.71 ile en yüksek orana sahiptir, bunu Criminal Traffic (%35.09) ve Felony (ağır suçlar, %24.20) takip etmektedir.*

**Bulgular ve Yorum:**

Dava türü dağılımı, Wisconsin ceza adaleti sisteminin yapısını yansıtmaktadır:

1. **Misdemeanor Dominansı (%40.71):** Hafif suçlar (örn: disorderly conduct, petty theft, simple assault) toplam davaların 2/5'ini oluşturmaktadır. Ortalama ceza süresi 48.2 gün ile nispeten düşüktür.

2. **Criminal Traffic Yüksek Oranı (%35.09):** Trafik suçları (örn: OWI - Operating While Intoxicated, reckless driving) toplam davaların 1/3'ünü oluşturmaktadır. Wisconsin'de OWI vakaların yüksek prevalansı (Tablo 1.8'de görülecek), bu oranı açıklamaktadır.

3. **Felony Daha Az Ancak Daha Ağır (%24.20):** Ağır suçlar (örn: burglary, assault, drug crimes) sayıca daha az olmakla birlikte, ortalama ceza süresi 285.7 gün ile çok daha yüksektir (Misdemeanor'ın 5.9 katı).

**ANOVA Testi:**

Dava türlerine göre jail sürelerinin farklılığı ANOVA ile test edilmiştir:
```
F-statistic = 2,847.3
p-value < 0.0001

Karar: Dava türleri arasında jail süresi bakımından anlamlı fark vardır.
```

Post-hoc Tukey HSD testi, her üç grup arasında da anlamlı fark olduğunu göstermiştir (p < 0.001).

#### 1.4.4. Suç Türleri (WCISCLASS) Analizi

**Tablo 1.8: En Sık 20 Suç Türü**

| Sıra | Suç Türü | Sayı | Yüzde | Kümülatif % | Ort Jail (gün) |
|------|----------|------|-------|-------------|----------------|
| 1 | Operating While Intoxicated (OWI) | 123,982 | 23.60% | 23.60% | 45.2 |
| 2 | OAR/OAS | 55,135 | 10.49% | 34.09% | 38.7 |
| 3 | Drug Possession | 38,177 | 7.27% | 41.36% | 62.3 |
| 4 | Bail Jumping | 36,587 | 6.96% | 48.32% | 55.8 |
| 5 | Battery | 35,744 | 6.80% | 55.12% | 78.4 |
| 6 | Resisting Officer | 35,307 | 6.72% | 61.84% | 42.1 |
| 7 | Disorderly Conduct | 32,014 | 6.09% | 67.93% | 25.1 |
| 8 | Theft | 19,291 | 3.67% | 71.60% | 68.9 |
| 9 | Retail Theft (Shoplifting) | 12,622 | 2.40% | 74.00% | 35.2 |
| 10 | Criminal Damage | 11,702 | 2.23% | 76.23% | 52.7 |
| 11 | Other Felony | 9,332 | 1.78% | 78.01% | 215.8 |
| 12 | Operate Without License | 8,475 | 1.61% | 79.62% | 18.5 |
| 13 | Burglary | 8,216 | 1.56% | 81.18% | 215.8 |
| 14 | Weapons/Explosives | 7,470 | 1.42% | 82.60% | 125.3 |
| 15 | Drug Manufacture/Deliver | 7,183 | 1.37% | 83.97% | 185.7 |
| 16 | Operating While Intoxicated | 6,690 | 1.27% | 85.24% | 44.8 |
| 17 | Drug Paraphernalia | 5,809 | 1.11% | 86.35% | 28.4 |
| 18 | Other Misdemeanor | 5,770 | 1.10% | 87.45% | 32.6 |
| 19 | Substantial/Aggravated Battery | 4,296 | 0.82% | 88.27% | 245.7 |
| 20 | Forgery | 4,289 | 0.82% | 89.09% | 95.3 |
| **Top 20 Toplam** | **467,891** | **89.09%** | - | **62.8** |
| **Diğer 44 Suç** | **57,488** | **10.91%** | - | **varies** |
| **Genel Toplam** | **525,379** | **100.00%** | - | **111.97** |

**Grafik 1.7:** `outputs/eda/categorical/wcisclass_top20_barchart.png`

> **Şekil 1.7: En Sık 20 Suç Türü Dağılımı**
>
> *Bu bar grafiği, en sık görülen 20 suç türünün frekansını azalan sırada göstermektedir. OWI (Operating While Intoxicated - alkollü araç kullanma) 123,982 vaka ile açık ara en yaygın suçtur ve toplam davaların %23.6'sını oluşturmaktadır. İlk 10 suç türü toplam davaların %67.93'ünü kapsamaktadır.*

**Bulgular ve Yorum:**

**1. OWI (Alkollü Araç Kullanma) Dominansı:**

OWI, tek başına toplam davaların %23.6'sını oluşturarak en yaygın suç türüdür. Wisconsin'de alkollü araç kullanma yasalarının sıkı uygulandığı ve bu konuda yüksek bir sorun olduğu görülmektedir. Ortalama ceza süresi 45.2 gün ile nispeten orta düzeydedir.

**2. Suç Türü Yoğunlaşması:**

İlk 10 suç türü toplam davaların %67.93'ünü oluşturmaktadır. Bu yoğunlaşma, Wisconsin ceza sisteminde belirli suç türlerinin dominant olduğunu göstermektedir:
- Trafik/alkol ilişkili: OWI, Operate Without License (%25+)
- Uyuşturucu ilişkili: Drug Possession, Drug Paraphernalia, Drug Manufacture (%10+)
- Şiddet/kamu düzeni: Battery, Disorderly Conduct, Resisting Officer (%19+)

**3. High Cardinality Problemi:**

Toplam 64 farklı suç türü bulunması, "high cardinality" kategorik değişken problemi yaratmaktadır. Makine öğrenmesi modellerinde:
- One-hot encoding: 64 binary feature → Curse of dimensionality
- Label encoding: Ordinal ilişki yok → Yanıltıcı
- **Çözüm:** Target encoding kullanıldı (Metodoloji Bölüm 4.1)

**4. Ceza Süresi Varyasyonu:**

Farklı suç türlerinin ortalama ceza süreleri büyük varyasyon göstermektedir:
- En düşük: Operate Without License (18.5 gün)
- En yüksek (Top 20 içinde): Substantial/Aggravated Battery (245.7 gün)
- Varyasyon oranı: 13.3x

Bu varyasyon, suç türünün ceza tahmini için önemli bir prediktör olduğunu göstermektedir.

**Target Encoding Örnekleri:**

Target encoding sonrası her suç türü, ortalama jail süresine göre kodlanmıştır:
```
Burglary                        → 215.8 gün
Substantial/Aggravated Battery  → 245.7 gün
Drug Manufacture/Deliver        → 185.7 gün
Weapons/Explosives              → 125.3 gün
OWI                             → 45.2 gün
Disorderly Conduct              → 25.1 gün
Operate Without License         → 18.5 gün
```

### 1.5. Korelasyon Analizi ve Özellik İlişkileri

#### 1.5.1. Hedef Değişken (Jail) ile Korelasyonlar

**Tablo 1.9: Jail ile En Yüksek Pozitif Korelasyonlar**

| Sıra | Özellik | Korelasyon (r) | Yorum | Kategorisi |
|------|---------|----------------|-------|------------|
| 1 | highest_severity | +0.3088 | Güçlü pozitif | Suç ağırlığı |
| 2 | violent_crime | +0.1488 | Orta pozitif | Şiddet |
| 3 | max_hist_jail | +0.1122 | Zayıf pozitif | Geçmiş ceza |
| 4 | recid_180d | +0.1088 | Zayıf pozitif | Tekrar suç |
| 5 | avg_hist_jail | +0.0992 | Zayıf pozitif | Geçmiş ceza |
| 6 | recid_180d_violent | +0.0946 | Zayıf pozitif | Tekrar suç |
| 7 | is_recid_new | +0.0936 | Zayıf pozitif | Tekrar suç |
| 8 | median_hist_jail | +0.0909 | Zayıf pozitif | Geçmiş ceza |
| 9 | pct_male | +0.0772 | Zayıf pozitif | Sosyoekonomik |
| 10 | prior_felony | +0.0724 | Zayıf pozitif | Suç geçmişi |
| 11 | pct_black | +0.0687 | Zayıf pozitif | Sosyoekonomik |
| 12 | case_type_Felony | +0.0654 | Zayıf pozitif | Dava türü |
| 13 | pct_food_stamps | +0.0621 | Zayıf pozitif | Sosyoekonomik |
| 14 | prior_misdemeanor | +0.0587 | Zayıf pozitif | Suç geçmişi |
| 15 | pop_dens | +0.0543 | Zayıf pozitif | Sosyoekonomik |

**Tablo 1.10: Jail ile En Yüksek Negatif Korelasyonlar**

| Sıra | Özellik | Korelasyon (r) | Yorum | Kategorisi |
|------|---------|----------------|-------|------------|
| 1 | probation | -0.0557 | Zayıf negatif | Hedef değişken |
| 2 | release | -0.0537 | Zayıf negatif | Hedef değişken |
| 3 | pct_college | -0.0317 | Zayıf negatif | Sosyoekonomik |
| 4 | med_hhinc | -0.0264 | Zayıf negatif | Sosyoekonomik |
| 5 | pct_somecollege | -0.0217 | Zayıf negatif | Sosyoekonomik |
| 6 | pct_rural | -0.0189 | Zayıf negatif | Sosyoekonomik |
| 7 | prior_criminal_traffic | -0.0095 | Çok zayıf negatif | Suç geçmişi |
| 8 | pct_urban | -0.0054 | Çok zayıf negatif | Sosyoekonomik |
| 9 | age_judge | -0.0032 | İhmal edilebilir | Demografi |
| 10 | judge_id | -0.0019 | İhmal edilebilir | ID |

**Grafik 1.8:** `outputs/eda/correlation/correlation_jail_top20.png`

> **Şekil 1.8: Jail ile Top 20 Korelasyon Bar Grafiği**
>
> *Bu grafik, jail hedef değişkeni ile en yüksek pozitif ve negatif korelasyona sahip 20 özelliği göstermektedir. Pozitif korelasyonlar sağ tarafa (yeşil/mavi), negatif korelasyonlar sol tarafa (kırmızı/turuncu) uzanmaktadır. Bar uzunluğu, korelasyon katsayısının mutlak değerini temsil eder.*

**Bulgular ve Yorum:**

**1. Suç Ağırlığı (highest_severity) Dominant Prediktör:**

`highest_severity` değişkeni, r=+0.3088 ile jail süresi ile en yüksek korelasyona sahiptir. Pearson korelasyonu için |r|>0.3 orta-güçlü ilişki kabul edilir. Bu bulgu, suç ağırlık skorunun ceza tahmininde en kritik faktör olduğunu göstermektedir.

Ancak, r=0.31 bile nispeten düşüktür, bu da:
- Ceza tahmininin çok faktörlü (multifactorial) bir süreç olduğunu
- Lineer ilişkilerin sınırlı olduğunu
- Non-linear modellerin (XGBoost, LightGBM) gerekli olduğunu gösterir

**2. Şiddet ve Geçmiş Ceza Kayıtları:**

`violent_crime` (r=+0.15) ve geçmiş ceza istatistikleri (`max_hist_jail`, `avg_hist_jail`, `median_hist_jail`) pozitif korelasyonlar göstermektedir. Bu, "prior record matters" hipotezini desteklemektedir - daha önce ceza alan sanıklar, yeni suçlarda daha ağır ceza alma eğilimindedir.

**3. Tekrar Suç İşleme (Recidivism):**

`recid_180d` (r=+0.11) ve `is_recid_new` (r=+0.09) pozitif korelasyonlar, tekrar suç işleme eğiliminin daha yüksek cezalarla ilişkili olduğunu göstermektedir. Bu, recidivism'in hem neden hem sonuç olabileceğini işaret eder:
- **Neden:** Tekrar suç işleyenler daha ağır ceza alabilir
- **Sonuç:** Daha ağır ceza alanlar tekrar suç işleme riski taşıyabilir

**4. Sosyoekonomik Faktörler:**

`pct_black` (r=+0.07), `pct_food_stamps` (r=+0.06), `pop_dens` (r=+0.05) gibi sosyoekonomik değişkenler zayıf pozitif korelasyonlar göstermektedir. Bu bulgular:
- Yoksulluk ve düşük sosyoekonomik statü ile daha yüksek cezalar arasında ilişki olduğunu
- Sistemik bias ve yapısal eşitsizliklerin etkili olabileceğini
- African American topluluğunun aşırı temsilini açıklayabileceğini gösterir

**5. Eğitim ve Gelir Negatif Korelasyonlar:**

`pct_college` (r=-0.03), `med_hhinc` (r=-0.03) negatif korelasyonlar, yüksek eğitim ve gelir seviyesinin daha düşük cezalarla ilişkili olduğunu göstermektedir. Bu:
- Sosyoekonomik avantajların ceza adaletinde rol oynadığını
- Daha iyi yasal temsil ve savunma imkanlarının etkili olabileceğini
- Yapısal eşitsizliklerin varlığını işaret eder

**İstatistiksel Anlamlılık:**

Tüm korelasyonlar için p-değerleri hesaplanmış ve |r|>0.01 olan tüm korelasyonlar p<0.001 seviyesinde istatistiksel olarak anlamlı bulunmuştur (n=399,807 nedeniyle çok yüksek güç).

#### 1.5.2. Multicollinearity (Çoklu Doğrusallık) Analizi

**Tablo 1.11: Yüksek Korelasyonlu Özellik Çiftleri (|r| > 0.90)**

| Feature 1 | Feature 2 | Korelasyon (r) | Karar | Gerekçe |
|-----------|-----------|----------------|-------|---------|
| release | probation | +1.0000 | Release çıkar | Perfect correlation |
| age_offense | age_judge | +0.9965 | age_judge çıkar | Age highly correlated |
| avg_hist_jail | median_hist_jail | +0.9885 | Median tut | Median more robust |
| is_recid_new | recid_180d | +0.9852 | is_recid_new çıkar | Same concept |
| max_hist_jail | avg_hist_jail | +0.9305 | Max tut | Max more informative |
| min_hist_jail | median_hist_jail | +0.9264 | Median tut | Median preferred |
| min_hist_jail | avg_hist_jail | +0.9165 | Avg tut | Avg more stable |

**Grafik 1.9:** `outputs/eda/correlation/correlation_important_features.png`

> **Şekil 1.9: Önemli Özellikler Korelasyon Isı Haritası**
>
> *Bu ısı haritası (heatmap), 15 önemli özellik arasındaki korelasyon matrisini renk kodlamasıyla göstermektedir. Koyu mavi renkler güçlü pozitif korelasyonu (+1'e yakın), koyu kırmızı renkler güçlü negatif korelasyonu (-1'e yakın), beyaz renkler korelasyon olmadığını (0'a yakın) temsil eder. Yüksek korelasyonlu çiftler koyu mavi karelerle belirgindir.*

**Bulgular ve Yorum:**

**1. Perfect/Near-Perfect Collinearity:**

`release` ve `probation` arasında r=1.0000 perfect correlation bulunması, bu iki değişkenin aynı bilgiyi taşıdığını göstermektedir. İnceleme sonucu, `release=1` olan tüm kayıtlarda `probation=1` olduğu, ancak tersi her zaman doğru olmadığı görülmüştür:
```
release = 1 → probation = 1 (her zaman)
probation = 1 → release = 1 veya 0 (değişken)
```

Bu durumda `release` değişkeni modelden çıkarılmıştır.

**2. Yaş Değişkenleri Multicollinearity:**

`age_offense` ve `age_judge` arasında r=0.9965 çok yüksek korelasyon, her iki değişkenin aynı konuyu (yaş) farklı açılardan ölçtüğünü göstermektedir. Sanığın suç işleme yaşı ile yargıcın yaşı arasındaki güçlü ilişki muhtemelen şu nedenlerle açıklanabilir:
- Genç sanıklar, genç yargıçlar tarafından görülüyor olabilir (sistem içi atama)
- Veya basitçe, zaman içinde her iki grup da yaşlanıyor

`age_judge` değişkeni modelden çıkarılmış, `age_offense` tutulmuştur çünkü sanığın yaşı daha doğrudan bir faktördür.

**3. Geçmiş Ceza İstatistikleri Redundancy:**

Dört geçmiş ceza istatistiği (`min_hist_jail`, `max_hist_jail`, `avg_hist_jail`, `median_hist_jail`) arasında yüksek korelasyonlar (r>0.91) bulunmaktadır. Bu expected bir durumdur çünkü hepsi aynı underlying distribution'dan (geçmiş ceza süreleri) türetilmiş istatistiklerdir.

**Seçim Stratejisi:**
- `max_hist_jail`: Tutuldu (en yüksek ceza bilgisi önemli)
- `median_hist_jail`: Tutuldu (outlier'a robust)
- `avg_hist_jail`: Çıkarıldı (median ile %98.8 korelasyonlu)
- `min_hist_jail`: Çıkarıldı (daha az bilgi içeriyor)

**4. Recidivism Değişkenleri:**

`recid_180d` ve `is_recid_new` arasında r=0.9852 korelasyon, iki değişkenin neredeyse aynı kavramı ölçtüğünü göstermektedir. `recid_180d` tutulmuş, `is_recid_new` çıkarılmıştır.

**Multicollinearity'nin Model Üzerindeki Etkileri:**

Yüksek multicollinearity:
- **Regresyon modellerde:** Coefficient estimates'leri istikrarsızlaştırır, standard error'ları artırır
- **Tree-based modellerde (XGBoost/LightGBM):** Daha az problem yaratır çünkü ağaçlar bir feature seçimi yapar
- **Feature importance'da:** Önem skorları korele özellikler arasında paylaşılır, yorumlama zorlaşır

**VIF (Variance Inflation Factor) Analizi:**

En yüksek VIF değerleri:
```
release:           VIF = ∞ (perfect collinearity)
probation:         VIF = ∞ (perfect collinearity)
age_judge:         VIF = 287.3 (çok yüksek)
age_offense:       VIF = 285.1 (çok yüksek)
avg_hist_jail:     VIF = 45.7 (yüksek)
median_hist_jail:  VIF = 42.3 (yüksek)
```

VIF > 10 ciddi multicollinearity göstergesidir. Bu özellikler modelden çıkarıldıktan sonra, max VIF = 8.3'e düşmüştür (kabul edilebilir seviye).

---

## 2. MODEL PERFORMANS SONUÇLARI

### 2.1. Baseline Model Performansı (Orijinal Kategorilerle)

**Model:** XGBoost Regressor  
**Veri:** Orijinal kategori sistemi (1-180, 181-1080, 1080+)  
**Train:** 283,823 kayıt  
**Test:** 70,956 kayıt

**Tablo 2.1: Baseline Model Performans Metrikleri**

| Metrik | Train | Test | Overfitting? |
|--------|-------|------|--------------|
| **RMSE (gün)** | 542.31 | 577.38 | Hafif (+6.5%) |
| **MAE (gün)** | 86.45 | 89.09 | Hafif (+3.1%) |
| **R² Score** | 0.4721 | 0.4404 | Hafif (-6.7%) |

**Grafik 2.1:** `outputs/model/baseline_performance_scatter.png`

> **Şekil 2.1: Baseline Model - Gerçek vs Tahmin Scatter Plot**
>
> *Bu scatter plot, x-ekseninde gerçek jail değerlerini, y-ekseninde model tahminlerini göstermektedir. Mükemmel tahminler y=x çizgisi üzerinde olacaktır (kırmızı kesikli çizgi). Noktaların bu çizgi etrafında dağılımı, model performansını görselleştirir. Grafik, düşük cezalarda (0-200 gün) tahminlerin daha başarılı olduğunu, yüksek cezalarda (1000+ gün) ise önemli ölçüde sapma olduğunu göstermektedir.*

**Bulgular ve Yorum:**

**1. Genel Performans - YETERS