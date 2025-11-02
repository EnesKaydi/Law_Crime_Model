# TEZ PROJESİ - ADIM ADIM UYGULAMA REHBERİ
## Yapay Zeka Destekli Hukuk Asistanı - Tekrar Yapma Kılavuzu

**Öğrenci:** Muhammed Enes Kaydı  
**Tarih:** 2 Kasım 2025  

---

## 🎯 BU DOKÜMANI KULLANIM AMACI

Bu doküman, tez projesindeki tüm adımları sıfırdan tekrar yapmak için hazırlanmıştır.  
Her adımda hangi script'in çalıştırılacağı ve ne bekleyeceğiniz açıkça belirtilmiştir.

---

## 📋 ÖN KOŞULLAR

### Gerekli Dosyalar
1. ✅ Büyük veri seti: `wcld.csv` (1.5M satır, ~800MB)
   - Konum: `/Users/muhammedeneskaydi/Desktop/3.SINIF 2.DÖNEM/TEZ/TEZ FİNAL/wcld.csv`

### Gerekli Yazılımlar
1. ✅ Python 3.12+ (Virtual environment ile)
2. ✅ Gerekli kütüphaneler:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - xgboost
   - scikit-learn
   - shap

### Kurulum
```bash
# Virtual environment aktifleştir
source /Users/muhammedeneskaydi/PycharmProjects/LAW/.venv/bin/activate

# Gerekli kütüphaneleri yükle (henüz yüklenmediyse)
pip install pandas numpy matplotlib seaborn xgboost scikit-learn shap
```

---

## 📖 ADIM ADIM UYGULAMA

### ADIM 1: Büyük Veri Setini Yükleme ve İnceleme

**Script:** `01_veri_yukleme_inceleme.py`

**Ne Yapıyor:**
- 1.5M satırlık wcld.csv dosyasını yükler
- Veri boyutunu, kolon sayısını gösterir
- Eksik değer oranlarını hesaplar

**Nasıl Çalıştırılır:**
```bash
cd /Users/muhammedeneskaydi/PycharmProjects/LAW
python 01_veri_yukleme_inceleme.py
```

**Beklenen Çıktı:**
- Toplam satır: 1,476,967
- Kolon sayısı: 54
- Tam dolu satırlar: 357,452 (%24.20)
- Eksik verili satırlar: 1,119,515 (%75.80)

**Süre:** ~3-5 saniye

---

### ADIM 2: Temiz Veri Ayırma (Tüm Kolonlar Dolu)

**Script:** `02_temiz_veri_ayirma.py`

**Ne Yapıyor:**
- Tüm kolonları dolu olan (NaN içermeyen) satırları seçer
- `wcld_Tüm_Kolonlar_Dolu.csv` olarak kaydeder

**Nasıl Çalıştırılır:**
```bash
python 02_temiz_veri_ayirma.py
```

**Beklenen Çıktı:**
- Temiz satır sayısı: 357,452
- Dosya boyutu: ~267 MB
- Kayıt yeri: `wcld_Tüm_Kolonlar_Dolu.csv`

**Süre:** ~5-10 saniye

---

### ADIM 3: Eksik Verilerden %15 Örneklem Alma

**Script:** `03_eksik_veri_orneklem.py`

**Ne Yapıyor:**
- Eksik verili satırlardan rastgele %15 seçer
- `random_state=42` ile tekrarlanabilir örnekleme yapar
- `wcld_Eksik_Veri_Yuzde15.csv` olarak kaydeder

**Nasıl Çalıştırılır:**
```bash
python 03_eksik_veri_orneklem.py
```

**Beklenen Çıktı:**
- Eksik verili satırlar: 1,119,515
- Seçilen örneklem: 167,927 (%15)
- Dosya boyutu: ~125 MB
- Kayıt yeri: `wcld_Eksik_Veri_Yuzde15.csv`

**Süre:** ~5-10 saniye

---

### ADIM 4: Final Veri Seti Birleştirme

**Script:** `04_final_dataset_birlestirme.py`

**Ne Yapıyor:**
- Temiz veri (357K) + Eksik veri örneklemi (167K) birleştirir
- `wcld_Final_Dataset.csv` olarak kaydeder

**Nasıl Çalıştırılır:**
```bash
python 04_final_dataset_birlestirme.py
```

**Beklenen Çıktı:**
- Final satır sayısı: 525,379
- Temiz veri oranı: %68.04
- Eksik veri oranı: %31.96
- Hedef değişken doluluğu:
  - jail: %76.1
  - probation: %87.3
  - release: %100.0
- Kayıt yeri: `wcld_Final_Dataset.csv`

**Süre:** ~10-15 saniye

---

### ADIM 5: Veri Keşif Analizi (EDA)

#### ADIM 5.1: Temel İstatistikler

**Script:** `05_01_EDA_temel_istatistikler.py`

**Ne Yapıyor:**
- Veri tipleri analizi
- Eksik değer tablosu (her kolon için)
- Sayısal değişkenlerin özet istatistikleri
- Sonuçları `outputs/` klasörüne kaydeder

**Nasıl Çalıştırılır:**
```bash
python 05_01_EDA_temel_istatistikler.py
```

**Beklenen Çıktılar:**
- Konsol'da detaylı istatistikler
- `outputs/temel_istatistikler.txt` dosyası

**Süre:** ~5 saniye

---

#### ADIM 5.2: Hedef Değişken Dağılımları

**Script:** `05_02_EDA_hedef_degiskenler.py`

**Ne Yapıyor:**
- jail, probation, release dağılımlarını görselleştirir
- Histogram ve box plot grafikleri oluşturur
- Ceza kategorileri (Hafif/Orta/Ağır) analizi

**Nasıl Çalıştırılır:**
```bash
python 05_02_EDA_hedef_degiskenler.py
```

**Beklenen Çıktılar:**
- 6 adet grafik (PNG formatında)
- `outputs/graphs/` klasörüne kaydedilir

**Süre:** ~10-15 saniye

---

#### ADIM 5.3: Kategorik Değişken Analizleri

**Script:** `05_03_EDA_kategorik_degiskenler.py`

**Ne Yapıyor:**
- sex, race, case_type, violent_crime dağılımları
- wcisclass (suç türleri) - en sık 20 suç
- Bar chart ve pie chart grafikleri

**Nasıl Çalıştırılır:**
```bash
python 05_03_EDA_kategorik_degiskenler.py
```

**Beklenen Çıktılar:**
- 5-6 adet grafik
- Konsol'da frekans tabloları

**Süre:** ~10 saniye

---

#### ADIM 5.4: Korelasyon Analizleri

**Script:** `05_04_EDA_korelasyon.py`

**Ne Yapıyor:**
- Sayısal değişkenler arası korelasyon matrisi
- Heatmap görselleştirme
- Hedef değişkenlerle en yüksek korelasyonlu özellikler

**Nasıl Çalıştırılır:**
```bash
python 05_04_EDA_korelasyon.py
```

**Beklenen Çıktılar:**
- Korelasyon heatmap (PNG)
- En önemli korelasyonlar tablosu

**Süre:** ~15-20 saniye

---

#### ADIM 5.5: İleri Düzey Analizler

**Script:** `05_05_EDA_ileri_analiz.py`

**Ne Yapıyor:**
- Yaş vs ceza süresi ilişkisi
- Irk vs ceza süresi (bias analizi)
- Suç geçmişi vs yeni ceza
- Recidivism (tekrar suç) oranları

**Nasıl Çalıştırılır:**
```bash
python 05_05_EDA_ileri_analiz.py
```

**Beklenen Çıktılar:**
- 4-5 adet grafik
- İstatistiksel bulgular

**Süre:** ~15-20 saniye

---

### ADIM 6: Feature Engineering ve Encoding

**Script:** `09_Feature_Engineering_ve_Encoding.py`

**Ne Yapıyor:**
- Kategorik değişkenleri encode eder (Label, OneHot, Frequency)
- Multicollinearity yönetimi (4 çift kaldırıldı)
- Yeni feature'lar oluşturur (6 adet)
- Düşük korelasyonlu feature'ları temizler

**Nasıl Çalıştırılır:**
```bash
python 09_Feature_Engineering_ve_Encoding.py
```

**Beklenen Çıktı:**
- wcld_Processed_For_Model.csv (525,379 × 43 kolon = 41 feature + 2 target)
- Dosya boyutu: ~164 MB

**Süre:** ~20-30 saniye

---

### ADIM 7: Normalizasyon ve Train-Test Split

**Script:** `10_Normalizasyon_ve_Train_Test_Split.py`

**Ne Yapıyor:**
- StandardScaler normalizasyonu (mean=0, std=1)
- Stratified %80-20 split (ceza kategorilerine göre)
- Scaler objesini kaydeder (.pkl)

**Nasıl Çalıştırılır:**
```bash
python 10_Normalizasyon_ve_Train_Test_Split.py
```

**Beklenen Çıktı:**
- X_train.csv, X_test.csv, y_train.csv, y_test.csv
- scaler.pkl (deployment için)
- feature_names.txt
- Train: 283,823 kayıt, Test: 70,956 kayıt

**Süre:** ~15-20 saniye

---

### ADIM 8: XGBoost Model Eğitimi

**Script:** `11_XGBoost_Model_Egitimi.py`

**Ne Yapıyor:**
- XGBoost Regressor ile jail prediction modeli
- GridSearchCV ile hyperparameter tuning (243 kombinasyon, 3-fold CV)
- Model ve metadata kaydı (.pkl)
- Feature importance analizi

**Nasıl Çalıştırılır:**
```bash
python 11_XGBoost_Model_Egitimi.py
```

**Beklenen Çıktı:**
- xgboost_jail_model.pkl (eğitilmiş model)
- model_info.pkl (metadata)
- feature_importance.csv
- 3 adet görsel (importance, prediction vs actual, residuals)
- Test R² = 0.4404, MAE = 89.09 gün

**Süre:** ~4-6 dakika (GridSearchCV nedeniyle)

---

### ADIM 9: Detaylı Performans Değerlendirme

**Script:** `12_Detayli_Performans_Degerlendirme.py`

**Ne Yapıyor:**
- Kategori bazlı performans (Hafif/Orta/Ağır)
- Hata dağılım analizi
- Yüzdesel hata aralıkları
- En iyi/kötü tahminler
- Prediction confidence intervals

**Nasıl Çalıştırılır:**
```bash
python 12_Detayli_Performans_Degerlendirme.py
```

**Beklenen Çıktı:**
- 2 adet detaylı grafik (kategori performans, hata dağılımı)
- kategori_metrikleri.csv
- en_iyi_tahminler.csv, en_kotu_tahminler.csv
- Hafif ceza MAE: 47.42 gün (mükemmel!)

**Süre:** ~10-15 saniye

---

### ADIM 10: Model Explainability Analizi

**Script:** `13_Model_Explainability_Analizi.py`

**Ne Yapıyor:**
- XGBoost built-in feature importance (Weight, Gain, Cover)
- Permutation importance (10 repeats)
- Partial dependence plots (top 6 features)
- Individual prediction analysis (3 örnek vaka)
- Bias analizi (ırk ve cinsiyet)

**Nasıl Çalıştırılır:**
```bash
python 13_Model_Explainability_Analizi.py
```

**Beklenen Çıktı:**
- 4 adet görsel (importance, permutation, PD plots, individual)
- xgboost_feature_importance.csv
- permutation_importance.csv
- Top 3 önemli: highest_severity, pct_somecollege, med_hhinc

**Süre:** ~2-3 dakika (permutation importance nedeniyle)

---

## 📸 EKRAN GÖRÜNTÜLERİ ALMA

Her adımdan sonra:
1. ✅ Terminal çıktısını kaydet (Cmd+Shift+4 ile seçili alan)
2. ✅ Grafikleri zaten `outputs/graphs/` klasörüne kaydediliyor
3. ✅ `SONUCLAR.md` dosyasına ekle

---

## ⚠️ SORUN GİDERME

### Hata: "FileNotFoundError"
- Dosya yollarını kontrol et
- wcld.csv dosyasının doğru konumda olduğundan emin ol

### Hata: "MemoryError"
- Daha küçük bir örneklem kullanmayı dene
- Gereksiz programları kapat

### Grafik görünmüyor
- `outputs/graphs/` klasörünün varlığını kontrol et
- Script içinde `plt.savefig()` satırını kontrol et

---

**Son Güncelleme:** 2 Kasım 2025  
**Hazırlayan:** GitHub Copilot + Muhammed Enes Kaydı
