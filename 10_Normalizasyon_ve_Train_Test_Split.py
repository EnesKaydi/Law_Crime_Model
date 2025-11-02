"""
10_Normalizasyon_ve_Train_Test_Split.py

Bu script:
- İşlenmiş veriyi (`wcld_Processed_For_Model.csv`) yükler
- Hedef değişkenleri (jail, release) ayırır
- Feature'ları StandardScaler ile normalize eder
- Stratified train-test split yapar (80-20)
- Ceza kategorilerine göre stratification (class imbalance için)
- Train ve test setlerini kaydeder
- Scaler objesini kaydeder (.pkl formatında - model deployment için)
- Tüm adımları SONUCLAR.md'ye kaydeder

Kullanım:
    /Users/muhammedeneskaydi/PycharmProjects/LAW/.venv/bin/python 10_Normalizasyon_ve_Train_Test_Split.py

Notlar:
- StandardScaler: mean=0, std=1 yapacak (XGBoost için iyi)
- Stratified split: Ceza kategorileri dengelenmesi için (Hafif/Orta/Ağır)
- random_state=42: Tekrarlanabilirlik
- Scaler deployment için gerekli (production'da aynı normalizasyon uygulanmalı)
"""

import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Ayarlar ---
BASE_DIR = "/Users/muhammedeneskaydi/PycharmProjects/LAW"
PROCESSED_CSV = os.path.join(BASE_DIR, "wcld_Processed_For_Model.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_data")
SONUCLAR_PATH = os.path.join(BASE_DIR, "SONUCLAR.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ADIM 7: NORMALIZASYON VE TRAIN-TEST SPLIT")
print("=" * 70)

# --- Veri Yükleme ---
print(f"\n📂 İşlenmiş veri yükleniyor: {PROCESSED_CSV}")
df = pd.read_csv(PROCESSED_CSV)
print(f"✅ Veri yüklendi. Satır: {len(df):,}, Kolon: {len(df.columns)}")

# ===== 1. HEDEF DEĞİŞKENLERİ AYIR =====
print("\n" + "=" * 70)
print("1. HEDEF DEĞİŞKENLERİ AYIRMA")
print("=" * 70)

# Hedef değişkenler
target_vars = ['jail', 'release']
print(f"\n  🎯 Hedef değişkenler: {target_vars}")

# Jail değeri olmayan kayıtları çıkar (NaN veya 0)
print(f"  🔍 Jail değeri kontrol ediliyor...")
original_len = len(df)
df_valid = df[df['jail'].notna() & (df['jail'] > 0)].copy()
removed_len = original_len - len(df_valid)

print(f"  ✅ Jail değeri olan kayıtlar seçildi")
print(f"    • Orijinal: {original_len:,}")
print(f"    • Geçerli: {len(df_valid):,}")
print(f"    • Çıkarılan: {removed_len:,} (%{removed_len/original_len*100:.2f})")

# Hedef değişkenleri ayır
y = df_valid[target_vars].copy()
X = df_valid.drop(columns=target_vars)

print(f"\n  📊 X (Features): {X.shape}")
print(f"  📊 y (Targets): {y.shape}")

# ===== 2. CEZA KATEGORİLERİ OLUŞTURMA (STRATİFİCATİON İÇİN) =====
print("\n" + "=" * 70)
print("2. CEZA KATEGORİLERİ OLUŞTURMA (STRATIFICATION)")
print("=" * 70)

# Jail değerlerine göre kategoriler (EDA'da kullandığımız gibi)
def categorize_jail(val):
    if val <= 180:
        return 'Hafif'
    elif val <= 1080:
        return 'Orta'
    else:
        return 'Agir'

y['jail_category'] = y['jail'].apply(categorize_jail)

category_counts = y['jail_category'].value_counts()
print(f"\n  📊 Ceza Kategorileri Dağılımı:")
for cat, count in category_counts.items():
    pct = count / len(y) * 100
    print(f"    • {cat}: {count:,} (%{pct:.2f})")

# ===== 3. FEATURE NORMALİZASYONU =====
print("\n" + "=" * 70)
print("3. FEATURE NORMALİZASYONU (STANDARDSCALER)")
print("=" * 70)

print(f"\n  ⚙️ StandardScaler uygulanıyor...")
print(f"    • Tüm feature'lar mean=0, std=1 yapılacak")

scaler = StandardScaler()

# Sadece sayısal kolonları normalize et (zaten hepsi sayısal olmalı)
numeric_cols = X.select_dtypes(include=[np.number]).columns
print(f"    • Normalize edilecek kolon: {len(numeric_cols)}")

# Fit ve transform
X_scaled = scaler.fit_transform(X[numeric_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)

print(f"  ✅ Normalizasyon tamamlandı")
print(f"    • Örnek öncesi değerler: {X[numeric_cols[0]].head(3).values}")
print(f"    • Örnek sonrası değerler: {X_scaled_df[numeric_cols[0]].head(3).values}")

# ===== 4. TRAIN-TEST SPLIT =====
print("\n" + "=" * 70)
print("4. TRAIN-TEST SPLIT (STRATIFIED)")
print("=" * 70)

print(f"\n  🔀 Stratified split uygulanıyor...")
print(f"    • Train: %80")
print(f"    • Test: %20")
print(f"    • Stratify: jail_category (Hafif/Orta/Ağır)")
print(f"    • Random state: 42")

# Split yap
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y['jail_category']
)

print(f"\n  ✅ Split tamamlandı!")
print(f"    • X_train: {X_train.shape}")
print(f"    • X_test: {X_test.shape}")
print(f"    • y_train: {y_train.shape}")
print(f"    • y_test: {y_test.shape}")

# Kategori dağılımları kontrol et
print(f"\n  📊 Train set kategori dağılımı:")
train_cats = y_train['jail_category'].value_counts()
for cat, count in train_cats.items():
    pct = count / len(y_train) * 100
    print(f"    • {cat}: {count:,} (%{pct:.2f})")

print(f"\n  📊 Test set kategori dağılımı:")
test_cats = y_test['jail_category'].value_counts()
for cat, count in test_cats.items():
    pct = count / len(y_test) * 100
    print(f"    • {cat}: {count:,} (%{pct:.2f})")

# ===== 5. VERİLERİ KAYDET =====
print("\n" + "=" * 70)
print("5. VERİLERİ KAYDETME")
print("=" * 70)

print(f"\n  💾 Train ve test setleri kaydediliyor...")

# Train set
X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, 'y_train.csv'), index=False)

# Test set
X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, 'y_test.csv'), index=False)

print(f"  ✅ CSV dosyaları kaydedildi:")
print(f"    • {OUTPUT_DIR}/X_train.csv")
print(f"    • {OUTPUT_DIR}/y_train.csv")
print(f"    • {OUTPUT_DIR}/X_test.csv")
print(f"    • {OUTPUT_DIR}/y_test.csv")

# ===== 6. SCALER OBJESİNİ KAYDET =====
print("\n" + "=" * 70)
print("6. SCALER OBJESİNİ KAYDETME")
print("=" * 70)

scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
print(f"\n  💾 Scaler objesi kaydediliyor: {scaler_path}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"  ✅ Scaler kaydedildi!")
print(f"    • Deployment'ta aynı scaler kullanılacak")
print(f"    • Yeni veri gelince: scaler.transform(new_data)")

# ===== 7. FEATURE NAMES KAYDET =====
print("\n" + "=" * 70)
print("7. FEATURE İSİMLERİNİ KAYDETME")
print("=" * 70)

feature_names_path = os.path.join(OUTPUT_DIR, 'feature_names.txt')
print(f"\n  💾 Feature isimleri kaydediliyor: {feature_names_path}")

with open(feature_names_path, 'w') as f:
    for col in X_train.columns:
        f.write(f"{col}\n")

print(f"  ✅ {len(X_train.columns)} feature ismi kaydedildi")

# ===== 8. ÖZET İSTATİSTİKLER =====
print("\n" + "=" * 70)
print("8. ÖZET İSTATİSTİKLER")
print("=" * 70)

print(f"\n  📊 Final Veri Seti Özeti:")
print(f"    • Toplam veri: {len(df_valid):,} satır")
print(f"    • Feature sayısı: {X_train.shape[1]}")
print(f"    • Hedef değişken: 2 (jail, release)")
print(f"    • Train set: {len(X_train):,} (%80)")
print(f"    • Test set: {len(X_test):,} (%20)")
print(f"    • Normalizasyon: StandardScaler (mean=0, std=1)")
print(f"    • Stratification: jail_category (Hafif/Orta/Ağır)")

# Hedef değişken istatistikleri
print(f"\n  📊 Hedef Değişken İstatistikleri (Train):")
print(f"    • jail ortalama: {y_train['jail'].mean():.2f} gün")
print(f"    • jail median: {y_train['jail'].median():.2f} gün")
print(f"    • jail std: {y_train['jail'].std():.2f} gün")
print(f"    • jail min: {y_train['jail'].min():.0f} gün")
print(f"    • jail max: {y_train['jail'].max():.0f} gün")

# ===== 9. SONUCLAR.MD'YE EKLEME =====
print("\n" + "=" * 70)
print("9. SONUCLAR.MD GÜNCELLEME")
print("=" * 70)

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

md_lines = []
md_lines.append(f"\n## ADIM 7: NORMALİZASYON VE TRAIN-TEST SPLIT ✅\n")
md_lines.append(f"**Tarih:** {now}\n\n")

md_lines.append("### 📊 Veri Seti Özeti\n")
md_lines.append(f"- **Toplam veri:** {len(df_valid):,} satır (jail>0 olanlar)")
md_lines.append(f"- **Feature sayısı:** {X_train.shape[1]}")
md_lines.append(f"- **Hedef değişken:** 2 (jail, release)")
md_lines.append(f"- **Çıkarılan kayıt:** {removed_len:,} (jail=0 veya NaN)\n")

md_lines.append("### 🔀 Train-Test Split\n")
md_lines.append("```")
md_lines.append(f"Train Set:")
md_lines.append(f"  • X_train: {X_train.shape[0]:,} satır × {X_train.shape[1]} feature")
md_lines.append(f"  • y_train: {y_train.shape[0]:,} satır × {y_train.shape[1]-1} target (+1 category)")  # -1 çünkü category geçici
md_lines.append(f"  • Oran: %{len(X_train)/len(df_valid)*100:.1f}")
md_lines.append("")
md_lines.append(f"Test Set:")
md_lines.append(f"  • X_test: {X_test.shape[0]:,} satır × {X_test.shape[1]} feature")
md_lines.append(f"  • y_test: {y_test.shape[0]:,} satır × {y_test.shape[1]-1} target")
md_lines.append(f"  • Oran: %{len(X_test)/len(df_valid)*100:.1f}")
md_lines.append("```\n")

md_lines.append("### ⚙️ Normalizasyon\n")
md_lines.append("- **Yöntem:** StandardScaler (sklearn)")
md_lines.append("- **İşlem:** mean=0, std=1")
md_lines.append(f"- **Normalize edilen kolon:** {len(numeric_cols)}")
md_lines.append("- **Scaler kaydedildi:** `model_data/scaler.pkl` (deployment için)\n")

md_lines.append("### 🎯 Stratification (Class Imbalance Yönetimi)\n")
md_lines.append("Ceza kategorilerine göre stratified split uygulandı:\n")
md_lines.append("**Train Set:**")
md_lines.append("```")
for cat, count in train_cats.items():
    pct = count / len(y_train) * 100
    md_lines.append(f"• {cat}: {count:,} (%{pct:.2f})")
md_lines.append("```\n")
md_lines.append("**Test Set:**")
md_lines.append("```")
for cat, count in test_cats.items():
    pct = count / len(y_test) * 100
    md_lines.append(f"• {cat}: {count:,} (%{pct:.2f})")
md_lines.append("```\n")

md_lines.append("### 📊 Hedef Değişken İstatistikleri (Train)\n")
md_lines.append("**jail (Hapis Süresi - Gün):**")
md_lines.append("```")
md_lines.append(f"• Ortalama: {y_train['jail'].mean():.2f} gün")
md_lines.append(f"• Median: {y_train['jail'].median():.2f} gün")
md_lines.append(f"• Std Sapma: {y_train['jail'].std():.2f} gün")
md_lines.append(f"• Min: {y_train['jail'].min():.0f} gün")
md_lines.append(f"• Max: {y_train['jail'].max():.0f} gün")
md_lines.append("```\n")

md_lines.append("### 💾 Kaydedilen Dosyalar\n")
md_lines.append("```")
md_lines.append("model_data/")
md_lines.append("  ├── X_train.csv (train features)")
md_lines.append("  ├── X_test.csv (test features)")
md_lines.append("  ├── y_train.csv (train targets)")
md_lines.append("  ├── y_test.csv (test targets)")
md_lines.append("  ├── scaler.pkl (StandardScaler objesi)")
md_lines.append("  └── feature_names.txt (feature isimleri)")
md_lines.append("```\n")

md_lines.append("### ✅ Önemli Notlar\n")
md_lines.append("- ✅ Veri normalize edildi (XGBoost için optimal)")
md_lines.append("- ✅ Stratified split ile class imbalance dengelendi")
md_lines.append("- ✅ Scaler kaydedildi (deployment'ta kullanılacak)")
md_lines.append("- ✅ Feature names kaydedildi (model yorumlama için)")
md_lines.append("- ✅ Train/test setleri hazır → Model eğitimine başlanabilir!\n")

md_lines.append("---\n")

# Dosyaya ekle
with open(SONUCLAR_PATH, 'a', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"✅ SONUCLAR.md güncellendi: {SONUCLAR_PATH}")

print("\n" + "=" * 70)
print("✅ ADIM 7 TAMAMLANDI!")
print("=" * 70)
print(f"\n📌 Sonraki adım: XGBoost Model Eğitimi")
print(f"📌 Hazır dosyalar: {OUTPUT_DIR}/")
