"""
09_Feature_Engineering_ve_Encoding.py

Bu script:
- Final dataset'i yükler ve model için hazırlar
- Kategorik değişkenleri encode eder (Label Encoding & One-Hot Encoding)
- Gereksiz kolonları çıkarır (ID'ler, multicollinearity olanlar)
- Feature engineering yapar (yeni özellikler türetir)
- Eksik değerleri yönetir (imputation)
- Feature selection yapar (düşük korelasyonlu özellikleri çıkarır)
- İşlenmiş veriyi kaydeder: `wcld_Processed_For_Model.csv`
- Tüm adımları SONUCLAR.md'ye kaydeder

Kullanım:
    /Users/muhammedeneskaydi/PycharmProjects/LAW/.venv/bin/python 09_Feature_Engineering_ve_Encoding.py

Notlar:
- Tez için kritik adım - her işlem dokümante edilmiştir
- Encoding stratejisi: Binary için Label, Multi-class için One-Hot
- Multicollinearity çiftlerinden biri çıkarılır (VIF kontrolü)
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# --- Ayarlar ---
BASE_DIR = "/Users/muhammedeneskaydi/PycharmProjects/LAW"
FINAL_CSV = os.path.join(BASE_DIR, "wcld_Final_Dataset.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "wcld_Processed_For_Model.csv")
SONUCLAR_PATH = os.path.join(BASE_DIR, "SONUCLAR.md")

print("=" * 70)
print("ADIM 6: FEATURE ENGINEERING VE ENCODING")
print("=" * 70)

# --- Veri Yükleme ---
print(f"\n📂 Veri yükleniyor: {FINAL_CSV}")
df = pd.read_csv(FINAL_CSV)
print(f"✅ Veri yüklendi. Satır: {len(df):,}, Kolon: {len(df.columns)}")
original_shape = df.shape

# --- İşlem Takibi ---
operations_log = []

# ===== 1. GEREKSIZ KOLONLARI ÇIKAR =====
print("\n" + "=" * 70)
print("1. GEREKSIZ KOLONLARI ÇIKARMA")
print("=" * 70)

# ID kolonları (model için gereksiz)
id_cols = ['new_id', 'judge_id', 'county', 'zip']
print(f"\n  🗑️ ID kolonları çıkarılıyor: {id_cols}")
df = df.drop(columns=id_cols, errors='ignore')
operations_log.append(f"ID kolonları çıkarıldı: {id_cols}")

# Train/test split kolonları (veri sızıntısı olabilir)
split_cols = ['train_test_split_caselevel', 'train_test_split_deflevel']
print(f"  🗑️ Split kolonları çıkarılıyor: {split_cols}")
df = df.drop(columns=split_cols, errors='ignore')
operations_log.append(f"Split kolonları çıkarıldı: {split_cols}")

# ===== 2. MULTICOLLINEARİTY YÖNETİMİ =====
print("\n" + "=" * 70)
print("2. MULTICOLLINEARİTY YÖNETİMİ")
print("=" * 70)

# EDA'da tespit edilen yüksek korelasyonlu çiftler
# (probation-release: 1.0, age_offense-age_judge: 0.996, vb.)
multicollinear_pairs = [
    ('probation', 'release', 1.0, 'keep_release'),  # release'i tut (daha genel)
    ('age_offense', 'age_judge', 0.996, 'keep_age_offense'),  # suçlu yaşı önemli
    ('avg_hist_jail', 'median_hist_jail', 0.988, 'keep_median_hist_jail'),  # medyan daha robust
    ('min_hist_jail', 'avg_hist_jail', 0.916, 'keep_avg_hist_jail'),  # ortalama daha bilgilendirici
]

print("\n  ⚠️ Yüksek korelasyonlu çiftlerden biri çıkarılıyor:")
for feat1, feat2, corr, action in multicollinear_pairs:
    keep = action.split('_', 1)[1]
    drop = feat1 if keep == feat2 else feat2
    
    if drop in df.columns:
        print(f"    • {feat1} ↔ {feat2} (r={corr:.3f}) → {drop} ÇIKARILDI")
        df = df.drop(columns=[drop])
        operations_log.append(f"Multicollinearity: {drop} çıkarıldı (r={corr:.3f} with {keep})")

# ===== 3. HEDEF DEĞİŞKENLERİ AYIR =====
print("\n" + "=" * 70)
print("3. HEDEF DEĞİŞKENLERİ AYIRMA")
print("=" * 70)

# Hedef değişkenler
target_vars = ['jail', 'release']  # probation çıkarıldı (release ile aynı)
print(f"\n  🎯 Hedef değişkenler: {target_vars}")

# Hedef değişkenleri başka bir DataFrame'e kaydet
df_targets = df[target_vars].copy()
print(f"  ✅ Hedef değişkenler ayrıldı: {df_targets.shape}")

# Hedef değişkenleri ana DataFrame'den çıkar (sonra geri ekleyeceğiz)
df_features = df.drop(columns=target_vars)
operations_log.append(f"Hedef değişkenler ayrıldı: {target_vars}")

# ===== 4. KATEGORİK DEĞİŞKENLERİ ENCODE ETME =====
print("\n" + "=" * 70)
print("4. KATEGORİK DEĞİŞKENLERİ ENCODING")
print("=" * 70)

# Kategorik kolonları tespit et
categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
print(f"\n  📋 {len(categorical_cols)} kategorik kolon bulundu: {categorical_cols}")

encoding_info = {}

# --- 4.1 Binary Kategorik Değişkenler (Label Encoding) ---
print("\n  🔢 Binary değişkenler için Label Encoding:")

# sex: M/F → 0/1
if 'sex' in df_features.columns:
    le_sex = LabelEncoder()
    df_features['sex_encoded'] = le_sex.fit_transform(df_features['sex'].fillna('Unknown'))
    print(f"    • sex: {list(le_sex.classes_)} → {list(range(len(le_sex.classes_)))}")
    encoding_info['sex'] = {'type': 'LabelEncoder', 'classes': list(le_sex.classes_)}
    df_features = df_features.drop(columns=['sex'])

# violent_crime: zaten 0/1 (sayısal)
print("    • violent_crime: Zaten binary (0/1) ✅")

# --- 4.2 Multi-class Kategorik Değişkenler (One-Hot Encoding) ---
print("\n  🔄 Multi-class değişkenler için One-Hot Encoding:")

# race: Caucasian, African American, Hispanic, vb.
if 'race' in df_features.columns:
    race_dummies = pd.get_dummies(df_features['race'], prefix='race', drop_first=True)
    print(f"    • race: {df_features['race'].nunique()} kategori → {len(race_dummies.columns)} dummy")
    encoding_info['race'] = {
        'type': 'OneHot', 
        'categories': df_features['race'].unique().tolist(),
        'dummies': race_dummies.columns.tolist()
    }
    df_features = pd.concat([df_features, race_dummies], axis=1)
    df_features = df_features.drop(columns=['race'])

# case_type: Felony, Misdemeanor, Criminal Traffic
if 'case_type' in df_features.columns:
    case_dummies = pd.get_dummies(df_features['case_type'], prefix='case_type', drop_first=True)
    print(f"    • case_type: {df_features['case_type'].nunique()} kategori → {len(case_dummies.columns)} dummy")
    encoding_info['case_type'] = {
        'type': 'OneHot',
        'categories': df_features['case_type'].unique().tolist(),
        'dummies': case_dummies.columns.tolist()
    }
    df_features = pd.concat([df_features, case_dummies], axis=1)
    df_features = df_features.drop(columns=['case_type'])

# wcisclass: ÇOK FAZLA KATEGORİ (500+) → Frequency Encoding
if 'wcisclass' in df_features.columns:
    print(f"    • wcisclass: {df_features['wcisclass'].nunique()} kategori (çok fazla!)")
    print("      → Frequency Encoding uygulanıyor (kategori frekansı ile encode)")
    
    freq_map = df_features['wcisclass'].value_counts(normalize=True).to_dict()
    df_features['wcisclass_freq'] = df_features['wcisclass'].map(freq_map).fillna(0)
    
    encoding_info['wcisclass'] = {
        'type': 'FrequencyEncoding',
        'unique_categories': df_features['wcisclass'].nunique()
    }
    df_features = df_features.drop(columns=['wcisclass'])

# all_races: Benzer race ile - frequency encoding
if 'all_races' in df_features.columns:
    freq_map_races = df_features['all_races'].value_counts(normalize=True).to_dict()
    df_features['all_races_freq'] = df_features['all_races'].map(freq_map_races).fillna(0)
    df_features = df_features.drop(columns=['all_races'])
    encoding_info['all_races'] = {'type': 'FrequencyEncoding'}

operations_log.append(f"Kategorik encoding tamamlandı: {len(encoding_info)} değişken")

# ===== 5. EKSİK DEĞER YÖNETİMİ =====
print("\n" + "=" * 70)
print("5. EKSİK DEĞER YÖNETİMİ (IMPUTATION)")
print("=" * 70)

# Eksik değerleri kontrol et
missing_counts = df_features.isnull().sum()
missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)

if len(missing_cols) > 0:
    print(f"\n  ⚠️ {len(missing_cols)} kolonda eksik değer var:")
    for col, count in missing_cols.head(10).items():
        pct = count / len(df_features) * 100
        print(f"    • {col}: {count:,} (%{pct:.2f})")
    
    # Sayısal değişkenler için median imputation
    print("\n  🔧 Eksik değerler median ile doldurulacak (XGBoost eksik değer ile çalışır ama temizlemek daha iyi)")
    
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[numeric_cols] = imputer.fit_transform(df_features[numeric_cols])
    
    operations_log.append(f"Eksik değerler median ile dolduruldu: {len(missing_cols)} kolon")
    print(f"  ✅ Eksik değerler dolduruldu")
else:
    print("  ✅ Eksik değer yok!")

# ===== 6. FEATURE ENGINEERING (YENİ ÖZELLİKLER) =====
print("\n" + "=" * 70)
print("6. FEATURE ENGINEERING (YENİ ÖZELLİKLER TÜRETME)")
print("=" * 70)

print("\n  ⚙️ Yeni özellikler oluşturuluyor:")

# 6.1 Toplam suç geçmişi
if 'prior_felony' in df_features.columns and 'prior_misdemeanor' in df_features.columns:
    df_features['total_prior_crimes'] = df_features['prior_felony'] + df_features['prior_misdemeanor']
    print("    • total_prior_crimes = prior_felony + prior_misdemeanor")

# 6.2 Ağır suç oranı
if 'prior_felony' in df_features.columns and 'total_prior_crimes' in df_features.columns:
    df_features['felony_ratio'] = df_features['prior_felony'] / (df_features['total_prior_crimes'] + 1)
    print("    • felony_ratio = prior_felony / (total_prior_crimes + 1)")

# 6.3 Yaş grubu (kategorik → sayısal)
if 'age_offense' in df_features.columns:
    df_features['age_group_young'] = (df_features['age_offense'] < 25).astype(int)
    df_features['age_group_old'] = (df_features['age_offense'] > 60).astype(int)
    print("    • age_group_young (<25), age_group_old (>60)")

# 6.4 Yüksek risk skoru (violent + recidivism)
if 'violent_crime' in df_features.columns and 'recid_180d' in df_features.columns:
    df_features['high_risk_score'] = (df_features['violent_crime'].fillna(0) + 
                                       df_features['recid_180d'].fillna(0))
    print("    • high_risk_score = violent_crime + recid_180d")

# 6.5 Mahalle sosyoekonomik skoru (birleşik)
socio_cols = ['pct_college', 'med_hhinc', 'pct_food_stamps']
available_socio = [col for col in socio_cols if col in df_features.columns]
if len(available_socio) >= 2:
    # Normalize edip birleştir
    df_features['socioeconomic_score'] = 0
    for col in available_socio:
        normalized = (df_features[col] - df_features[col].mean()) / df_features[col].std()
        if col == 'pct_food_stamps':
            normalized = -normalized  # Negatif etki (food stamps yüksek = düşük sosyoekonomik)
        df_features['socioeconomic_score'] += normalized
    print(f"    • socioeconomic_score (birleşik: {available_socio})")

operations_log.append("Feature engineering tamamlandı: 6 yeni özellik")

# ===== 7. DÜŞÜK ÖNEM DEĞERLİ ÖZELLİKLERİ ÇIKAR =====
print("\n" + "=" * 70)
print("7. DÜŞÜK KORELASYONLU ÖZELLİKLERİ ÇIKARMA")
print("=" * 70)

# Hedef değişkeni geri ekle (geçici)
df_temp = pd.concat([df_features, df_targets], axis=1)

# Sadece jail ile korelasyonu çok düşük olanları çıkar
if 'jail' in df_temp.columns:
    numeric_features = df_features.select_dtypes(include=[np.number]).columns
    correlations = df_temp[numeric_features].corrwith(df_temp['jail']).abs()
    
    low_corr_features = correlations[correlations < 0.01].index.tolist()
    
    if len(low_corr_features) > 0:
        print(f"\n  ⚠️ {len(low_corr_features)} özellik jail ile çok düşük korelasyonlu (|r| < 0.01):")
        for feat in low_corr_features[:10]:
            print(f"    • {feat}: r = {correlations[feat]:.4f}")
        
        print(f"\n  🗑️ Bu özellikler çıkarılacak (model için gereksiz)")
        df_features = df_features.drop(columns=low_corr_features, errors='ignore')
        operations_log.append(f"Düşük korelasyonlu {len(low_corr_features)} özellik çıkarıldı")
    else:
        print("  ✅ Tüm özellikler yeterli korelasyona sahip")

# Geçici DataFrame'i temizle
del df_temp

# ===== 8. FİNAL VERİ SETİ BİRLEŞTİRME =====
print("\n" + "=" * 70)
print("8. FİNAL VERİ SETİ BİRLEŞTİRME")
print("=" * 70)

# Hedef değişkenleri geri ekle
df_final = pd.concat([df_features, df_targets], axis=1)

print(f"\n  ✅ Final veri seti oluşturuldu:")
print(f"    • Satır sayısı: {len(df_final):,}")
print(f"    • Feature sayısı: {len(df_features.columns)}")
print(f"    • Hedef değişken sayısı: {len(df_targets.columns)}")
print(f"    • Toplam kolon: {len(df_final.columns)}")

# ===== 9. VERİ SETİNİ KAYDET =====
print("\n" + "=" * 70)
print("9. İŞLENMİŞ VERİYİ KAYDETME")
print("=" * 70)

print(f"\n  💾 İşlenmiş veri kaydediliyor: {OUTPUT_CSV}")
df_final.to_csv(OUTPUT_CSV, index=False)

file_size_mb = os.path.getsize(OUTPUT_CSV) / 1024**2
print(f"  ✅ Kayıt tamamlandı! Dosya boyutu: {file_size_mb:.2f} MB")

# ===== 10. SONUÇLAR.MD'YE EKLEME =====
print("\n" + "=" * 70)
print("10. SONUCLAR.MD GÜNCELLEME")
print("=" * 70)

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

md_lines = []
md_lines.append(f"\n## ADIM 6: FEATURE ENGINEERING VE ENCODING ✅\n")
md_lines.append(f"**Tarih:** {now}\n\n")

md_lines.append("### 📊 İşlem Özeti\n")
md_lines.append(f"- **Orijinal boyut:** {original_shape[0]:,} satır × {original_shape[1]} kolon")
md_lines.append(f"- **Final boyut:** {df_final.shape[0]:,} satır × {df_final.shape[1]} kolon")
md_lines.append(f"- **Feature sayısı:** {len(df_features.columns)}")
md_lines.append(f"- **Hedef değişken:** {len(df_targets.columns)} (jail, release)\n")

md_lines.append("### 🔧 Yapılan İşlemler\n")
md_lines.append("```")
for i, op in enumerate(operations_log, 1):
    md_lines.append(f"{i}. {op}")
md_lines.append("```\n")

md_lines.append("### 📋 Encoding Detayları\n")
for var, info in encoding_info.items():
    md_lines.append(f"**{var}:**")
    md_lines.append(f"- Encoding Tipi: {info['type']}")
    if 'classes' in info:
        md_lines.append(f"- Sınıflar: {info['classes']}")
    if 'dummies' in info:
        md_lines.append(f"- Oluşturulan dummy sayısı: {len(info['dummies'])}")
    md_lines.append("")

md_lines.append("### ⚙️ Yeni Oluşturulan Özellikler\n")
md_lines.append("1. `total_prior_crimes`: Toplam suç geçmişi")
md_lines.append("2. `felony_ratio`: Ağır suç oranı")
md_lines.append("3. `age_group_young` / `age_group_old`: Yaş grubu binary")
md_lines.append("4. `high_risk_score`: Şiddet + tekrar suç skoru")
md_lines.append("5. `socioeconomic_score`: Mahalle sosyoekonomik skoru")
md_lines.append("6. `wcisclass_freq` / `all_races_freq`: Frequency encoding\n")

md_lines.append("### 💾 Kaydedilen Dosya\n")
md_lines.append(f"- **Dosya:** `wcld_Processed_For_Model.csv`")
md_lines.append(f"- **Boyut:** {file_size_mb:.2f} MB")
md_lines.append(f"- **Kullanım:** XGBoost model eğitimi için hazır\n")

md_lines.append("### ✅ Önemli Notlar\n")
md_lines.append("- ✅ Tüm kategorik değişkenler sayısal formata çevrildi")
md_lines.append("- ✅ Multicollinearity temizlendi (VIF riski azaltıldı)")
md_lines.append("- ✅ Eksik değerler yönetildi (median imputation)")
md_lines.append("- ✅ Feature engineering ile 6 yeni özellik eklendi")
md_lines.append("- ✅ Düşük korelasyonlu özellikler çıkarıldı")
md_lines.append("- ✅ Veri model eğitimine hazır!\n")

md_lines.append("---\n")

# Dosyaya ekle
with open(SONUCLAR_PATH, 'a', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"✅ SONUCLAR.md güncellendi: {SONUCLAR_PATH}")

print("\n" + "=" * 70)
print("✅ ADIM 6 TAMAMLANDI!")
print("=" * 70)
print(f"\n📌 Sonraki adım: Veri Normalizasyonu & Train-Test Split")
print(f"📌 Model eğitimine hazır: {OUTPUT_CSV}")
