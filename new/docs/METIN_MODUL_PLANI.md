# 🏛️ Metin Tabanlı Ceza Tahmin Modülü
## AI-LAW Text Processing Module

---

## 📋 Genel Bakış

Bu modül, kullanıcının girdiği metin tabanlı olay açıklamasını işleyerek:
1. Suç kategorisini tespit eder
2. TCK (Türk Ceza Kanunu) madde numarasını belirler
3. Ceza aralığını tahmin eder
4. Mevcut CatBoost modeli ile kesin ceza süresini hesaplar

---

## 🎯 Hedef

```
Kullanıcı Metni:
"Adamın biri markete girip 5000 TL değerinde ürün çaldı, güvenlik kamerası kaydı var"

↓
[MODÜL ÇIKTISI]

📌 Suç Kategorisi: Hırsızlık (Theft)
📌 TCK Madde: 141
📌 Ceza Aralığı: 1-3 Yıl Hapis
📌 Model Tahmini: 720-1095 Gün (2-3 Yıl)
```

---

## 🔧 Teknik Mimari

```
┌─────────────────────────────────────────────────────────────────┐
│                    METİN TABANLI MODÜL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. METİN GİRİŞİ                                              │
│     ├── Form input (textarea)                                    │
│     └── Örnek: "Kişi marketten hırsızlık yaptı"               │
│                         ↓                                       │
│  2. ÖN İŞLEME (Preprocessing)                                  │
│     ├── Türkçe karakter normalizasyonu                           │
│     ├── Küçük harf dönüşümü                                    │
│     └── Stopwords temizleme                                     │
│                         ↓                                       │
│  3. ANAHTAR KELİME EŞLEŞTİRME (Keyword Matching)             │
│     ├── Suç türü tespiti                                        │
│     ├── Şiddet faktörü                                          │
│     ├── Zarar miktarı                                            │
│     └── Tahmin edilen özellikler                                │
│                         ↓                                       │
│  4. TCK EŞLEŞTİRME                                             │
│     ├── Suç → TCK Madde eşleştirme                              │
│     └── Ceza aralığı belirleme                                  │
│                         ↓                                       │
│  5. FEATURE ÇIKARMA                                             │
│     ├── highest_severity                                         │
│     ├── violent_crime                                            │
│     ├── is_recid_new                                            │
│     └── Diğer özellikler                                        │
│                         ↓                                       │
│  6. CATBOOST MODEL                                              │
│     └── Ceza tahmini (gün)                                      │
│                         ↓                                       │
│  7. SONUÇ GÖSTERİMİ                                            │
│     ├── Suç kategorisi                                          │
│     ├── TCK Madde                                               │
│     ├── Ceza aralığı                                            │
│     └── Kesin tahmin                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 TCK Eşleştirme Tablosu

| # | Wisconsin Suç | Türkçe Karşılık | TCK Madde | Ceza Aralığı |
|---|---------------|-----------------|-----------|--------------|
| 1 | Theft | Hırsızlık | 141-142 | 1-3 yıl |
| 2 | Burglary | Bina İçi Hırsızlık | 142-143 | 2-5 yıl |
| 3 | Battery | Yaralama | 86-89 | 2-4 yıl |
| 4 | Assault | Kasten Yaralama | 86-87 | 1-3 yıl |
| 5 | Armed Robbery | Silahla Soygun | 149-150 | 5-10 yıl |
| 6 | Robbery | Yağma | 148-149 | 3-8 yıl |
| 7 | Drug Possession | Uyuşturucu Bulundurma | 188 | 2-5 yıl |
| 8 | Drug Sale | Uyuşturucu Ticareti | 188-190 | 5-10 yıl |
| 9 | Weapons | Silah Bulundurma | 174-176 | 1-3 yıl |
| 10 | Murder | Kasten Öldürme | 81-85 | 15-24 yıl |
| 11 | Rape | Cinsel Saldırı | 102-103 | 5-15 yıl |
| 12 | Child Abuse | Çocuk İstismarı | 103, 272-273 | 3-8 yıl |
| 13 | Fraud | Dolandırıcılık | 158-164 | 1-3 yıl |
| 14 | Forgery | Sahtecilik | 159-160 | 1-3 yıl |
| 15 | Extortion | Tehdid | 106-107 | 6 ay-2 yıl |

---

## 🔍 Anahtar Kelime Listesi

### Hırsızlık (Theft)
- çalmak, çaldı, hırsızlık, gaspetmek, zimmetine geçirmek
- market, dükkân, araba, ev, işyeri
- para, değer, eşya, ürün

### Yaralama (Battery/Assault)
- vurmak, yaralamak, dövmek, saldırmak
- darp, yaralama, müessir fiziksel temas

### Soygun (Robbery)
- silah, bıçak, tehdit
- zorla almak, gasp, soygun

### Uyuşturucu (Drug)
- uyuşturucu, hap, eroin, esrar, kokain
- satmak, bulundurmak, ticaret

### Silah (Weapons)
- silah, bıçak, tabanca, kesici alet
- bulundurmak, taşımak

---

## 📦 Çıktı Formatı

```json
{
  "success": true,
  "data": {
    "suç_kategorisi": "Hırsızlık",
    "suç_kategorisi_en": "Theft",
    "tck_madde": "141",
    "tck_madde_aciklama": "Hırsızlık",
    "ceza_araligi": "1-3 Yıl",
    "ceza_araligi_gun": "365-1095",
    "model_tahmini_gun": 730,
    "model_tahmini_yil": 2.0,
    "guven_orani": "%85",
    "ozellikler": {
      "highest_severity": 8,
      "violent_crime": 0,
      "is_recid_new": 0,
      "age_offense": 25
    }
  },
  "metadata": {
    "islem_suresi": "0.5s",
    "kaynak": "Wisconsin Model + TCK Eşleştirme"
  }
}
```

---

## ⚙️ Kurulum Gereksinimleri

```bash
pip install -r requirements.txt
```

Gerekli kütüphaneler:
- pandas
- numpy
- catboost
- scikit-learn
- streamlit (frontend)

---

## 🚀 Kullanım

### Terminal
```python
from text_predictor import TextPredictor

predictor = TextPredictor()
sonuc = predictor.predict("Marketten 5000 TL değerinde ürün çaldı")
print(sonuc)
```

### Streamlit
```bash
streamlit run app.py
```

---

## 📁 Dosya Yapısı

```
new/
├── docs/
│   └── METIN_MODUL_PLANI.md    ← Bu dosya
├── data/
│   ├── tck_veritabani.csv      ← TCK eşleştirme tablosu
│   └── anahtar_kelimeler.json   ← Keyword listeleri
├── src/
│   ├── __init__.py
│   ├── text_preprocessor.py     ← Metin ön işleme
│   ├── keyword_matcher.py      ← Anahtar kelime eşleştirme
│   ├── feature_extractor.py   ← Feature çıkarma
│   ├── tck_mapper.py          ← TCK eşleştirme
│   └── predictor.py            ← Ana tahmin sınıfı
├── models/
│   └── (eğitilmiş modeller buraya)
├── tests/
│   └── test_predictor.py
├── app.py                      ← Streamlit UI
└── main.py                     ← Ana giriş noktası
```

---

## 📅 Geliştirme Aşamaları

| Aşama | Görev | Süre |
|-------|-------|------|
| 1 | TCK veritabanı oluşturma | 1 gün |
| 2 | Keyword matching sistemi | 2 gün |
| 3 | Feature çıkarıcı | 1 gün |
| 4 | Model entegrasyonu | 1 gün |
| 5 | Streamlit UI | 1 gün |
| 6 | Test & Optimizasyon | 2 gün |

**Toplam: ~8 gün**

---

## ✅ Başarı Kriterleri

1. En az 10 farklı suç türünü doğru tespit edebilmeli
2. TCK madde eşleştirmesi doğru olmalı
3. CatBoost modeli ile tahmin yapabilmeli
4. Kullanıcı dostu arayüz (Streamlit)
5. %80+ doğruluk hedefi

---

## 📝 Notlar

- İlk aşamada basit keyword matching kullanılacak
- İleride BERTurk entegrasyonu düşünülebilir
- Mevcut CatBoost modeli kullanılacak (yeniden eğitilmeyecek)
- GPU gerektirmez (MacBook M3 Pro yeterli)

---

**Son Güncelleme:** 4 Mart 2026
**Versiyon:** 1.0
**Hazırlayan:** Muhammed Enes Kaydı
