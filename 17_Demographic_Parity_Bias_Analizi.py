#!/usr/bin/env python3
"""
DEMOGRAPHIC PARITY & BIAS ANALİZİ

Bu script, modelin ırksal ve cinsiyet bazlı bias içerip içermediğini analiz eder.

Metrikler:
  1. Demographic Parity: Her grup için ortalama tahmin vs gerçek
  2. MAE per Group: Her grup için ortalama mutlak hata
  3. Equalized Odds: Her grup için hata dağılımları

TEZ için kritik etik bölüm!
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("DEMOGRAPHIC PARITY & BIAS ANALİZİ")
print("=" * 80)
print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Çıktı klasörü
output_dir = Path('outputs/bias_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. VERİ VE MODEL YÜKLEME
# ============================================================================
print("📂 Veri ve model yükleniyor...")

# Test verisi
X_test = pd.read_csv('model_data_new_categories/X_test.csv')
y_test = pd.read_csv('model_data_new_categories/y_test.csv')

# Orijinal veriyi yükle (race ve sex bilgisi için)
df_original = pd.read_csv('wcld_Final_Dataset.csv', low_memory=False)

# Model
with open('outputs/new_categories/new_category_model.pkl', 'rb') as f:
    model_info = pickle.load(f)
    model = model_info['model']

print(f"   Test kayıtları: {len(X_test):,}")
print(f"   Model: 3 Kategori (1-60, 61-365, 366+)\n")

# ============================================================================
# 2. TAHMİN YAP
# ============================================================================
print("🤖 Tahminler yapılıyor...")
y_pred = model.predict(X_test)
print("   ✅ Tahminler tamamlandı!\n")

# ============================================================================
# 3. RACE VE SEX BİLGİLERİNİ EKLE
# ============================================================================
print("🔍 Race ve Sex bilgileri ekleniyor...")

# Test verisinin indekslerini al (orijinal veriyle eşleştirmek için)
# Not: Bu biraz hack ama çalışıyor
# Daha iyi yöntem: jail sürelerine göre match yap

# Basit yöntem: jail değerine göre merge
df_test = y_test.copy()
df_test['y_pred'] = y_pred

# Orijinal veriden race ve sex al
if 'race' in df_original.columns and 'sex' in df_original.columns:
    # jail değerine göre merge (approximate)
    # Not: Bu ideal değil ama hızlı bir çözüm
    df_test = df_test.reset_index(drop=True)
    
    # Alternatif: Random sample from original (şimdilik bu yöntemi kullanalım)
    # Daha doğru olması için index'leri saklamak gerekir
    
    # DAHA İYİ YÖNTEM: Processed veriyi tekrar yükle (eğer race/sex varsa)
    df_processed = pd.read_csv('wcld_Processed_For_Model.csv', low_memory=False)
    
    if 'race' in df_processed.columns and 'sex' in df_processed.columns:
        # jail > 0 olanları filtrele
        df_processed = df_processed[df_processed['jail'] > 0].reset_index(drop=True)
        
        # Train-test split'i tekrarla (aynı random_state ile)
        from sklearn.model_selection import train_test_split
        
        def categorize_3(jail):
            if jail <= 60:
                return 'Hafif'
            elif jail <= 365:
                return 'Orta'
            else:
                return 'Agir'
        
        df_processed['jail_category_3'] = df_processed['jail'].apply(categorize_3)
        
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['jail', 'release', 'probation', 'jail_category', 
                                      'jail_category_new', 'jail_category_3', 'race', 'sex']]
        
        X_full = df_processed[feature_cols]
        y_full = df_processed['jail']
        categories_full = df_processed['jail_category_3']
        race_full = df_processed['race']
        sex_full = df_processed['sex']
        
        # Aynı split
        _, X_test_check, _, y_test_check, _, _, _, race_test, _, sex_test = train_test_split(
            X_full, y_full, categories_full, race_full, sex_full,
            test_size=0.2,
            random_state=42,
            stratify=categories_full
        )
        
        df_test['race'] = race_test.values
        df_test['sex'] = sex_test.values
        
        print(f"   ✅ Race ve Sex bilgileri eklendi!")
        print(f"   Unique races: {df_test['race'].nunique()}")
        print(f"   Unique sexes: {df_test['sex'].nunique()}\n")
        
    else:
        print("   ⚠️  Race/Sex bilgisi processed veride bulunamadı!")
        print("   ℹ️  Simüle edilmiş analiz yapılacak...\n")
        # Simüle et (eğer veri yoksa)
        df_test['race'] = np.random.choice(['Caucasian', 'African American', 'Hispanic'], len(df_test))
        df_test['sex'] = np.random.choice(['M', 'F'], len(df_test))
        
else:
    print("   ⚠️  Race/Sex bilgisi orijinal veride bulunamadı!")
    df_test['race'] = 'Unknown'
    df_test['sex'] = 'Unknown'

# ============================================================================
# 4. DEMOGRAPHIC PARITY ANALİZİ
# ============================================================================
print("📊 Demographic Parity Analizi...")
print("-" * 80)

# RACE BAZLI
print("\n🌍 IRKA GÖRE PERFORMANS:")
race_results = []
for race in df_test['race'].unique():
    if race == 'Unknown':
        continue
    mask = df_test['race'] == race
    n = mask.sum()
    
    if n < 10:  # Çok az veri varsa atla
        continue
    
    y_true = df_test.loc[mask, 'jail']
    y_pred_race = df_test.loc[mask, 'y_pred']
    
    mae = np.abs(y_true - y_pred_race).mean()
    mean_true = y_true.mean()
    mean_pred = y_pred_race.mean()
    
    race_results.append({
        'Race': race,
        'N': n,
        'Ort_Gerçek': mean_true,
        'Ort_Tahmin': mean_pred,
        'Fark': mean_pred - mean_true,
        'MAE': mae
    })

df_race = pd.DataFrame(race_results).sort_values('Ort_Gerçek', ascending=False)
print(df_race.to_string(index=False))

# GENDER BAZLI
print("\n\n👤 CİNSİYETE GÖRE PERFORMANS:")
gender_results = []
for gender in df_test['sex'].unique():
    if gender == 'Unknown':
        continue
    mask = df_test['sex'] == gender
    n = mask.sum()
    
    if n < 10:
        continue
    
    y_true = df_test.loc[mask, 'jail']
    y_pred_gender = df_test.loc[mask, 'y_pred']
    
    mae = np.abs(y_true - y_pred_gender).mean()
    mean_true = y_true.mean()
    mean_pred = y_pred_gender.mean()
    
    gender_results.append({
        'Gender': gender,
        'N': n,
        'Ort_Gerçek': mean_true,
        'Ort_Tahmin': mean_pred,
        'Fark': mean_pred - mean_true,
        'MAE': mae
    })

df_gender = pd.DataFrame(gender_results).sort_values('Ort_Gerçek', ascending=False)
print(df_gender.to_string(index=False))

# CSV kaydet
df_race.to_csv(output_dir / 'race_bias_analysis.csv', index=False)
df_gender.to_csv(output_dir / 'gender_bias_analysis.csv', index=False)

# ============================================================================
# 5. FAIRNESS METRİKLERİ
# ============================================================================
print("\n\n📊 FAIRNESS METRİKLERİ:")
print("-" * 80)

# Irk bazlı fairness
if len(df_race) > 1:
    max_pred = df_race['Ort_Tahmin'].max()
    min_pred = df_race['Ort_Tahmin'].min()
    fairness_ratio_race = min_pred / max_pred if max_pred > 0 else 0
    
    print(f"\n🌍 Irk Bazlı Fairness:")
    print(f"   En yüksek ort tahmin: {max_pred:.1f} gün ({df_race.loc[df_race['Ort_Tahmin'].idxmax(), 'Race']})")
    print(f"   En düşük ort tahmin: {min_pred:.1f} gün ({df_race.loc[df_race['Ort_Tahmin'].idxmin(), 'Race']})")
    print(f"   Fairness Ratio: {fairness_ratio_race:.3f} (1.0 = mükemmel eşitlik)")
    
    if fairness_ratio_race < 0.80:
        print("   ⚠️  Fairness ratio < 0.80: Potansiyel bias var!")
    else:
        print("   ✅ Fairness ratio >= 0.80: Kabul edilebilir seviye")

# Cinsiyet bazlı fairness
if len(df_gender) > 1:
    max_pred_g = df_gender['Ort_Tahmin'].max()
    min_pred_g = df_gender['Ort_Tahmin'].min()
    fairness_ratio_gender = min_pred_g / max_pred_g if max_pred_g > 0 else 0
    
    print(f"\n👤 Cinsiyet Bazlı Fairness:")
    print(f"   En yüksek ort tahmin: {max_pred_g:.1f} gün ({df_gender.loc[df_gender['Ort_Tahmin'].idxmax(), 'Gender']})")
    print(f"   En düşük ort tahmin: {min_pred_g:.1f} gün ({df_gender.loc[df_gender['Ort_Tahmin'].idxmin(), 'Gender']})")
    print(f"   Fairness Ratio: {fairness_ratio_gender:.3f}")
    
    if fairness_ratio_gender < 0.80:
        print("   ⚠️  Fairness ratio < 0.80: Potansiyel bias var!")
    else:
        print("   ✅ Fairness ratio >= 0.80: Kabul edilebilir seviye")

# ============================================================================
# 6. GÖRSELLEŞTİRME
# ============================================================================
print("\n\n📊 Grafikler oluşturuluyor...")

# Grafik 1: Irk bazlı ortalama tahmin vs gerçek
if len(df_race) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_race))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_race['Ort_Gerçek'], width, label='Gerçek Ortalama', color='skyblue')
    bars2 = ax.bar(x + width/2, df_race['Ort_Tahmin'], width, label='Tahmin Ortalama', color='lightcoral')
    
    ax.set_xlabel('Irk', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ortalama Ceza Süresi (gün)', fontsize=12, fontweight='bold')
    ax.set_title('Irk Bazlı Bias Analizi - Ortalama Ceza Süreleri', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_race['Race'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Değerleri ekle
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'race_bias_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_dir / 'race_bias_comparison.png'}")
    plt.close()

# Grafik 2: Cinsiyet bazlı
if len(df_gender) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_gender))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_gender['Ort_Gerçek'], width, label='Gerçek Ortalama', color='skyblue')
    bars2 = ax.bar(x + width/2, df_gender['Ort_Tahmin'], width, label='Tahmin Ortalama', color='lightcoral')
    
    ax.set_xlabel('Cinsiyet', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ortalama Ceza Süresi (gün)', fontsize=12, fontweight='bold')
    ax.set_title('Cinsiyet Bazlı Bias Analizi - Ortalama Ceza Süreleri', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_gender['Gender'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gender_bias_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_dir / 'gender_bias_comparison.png'}")
    plt.close()

# ============================================================================
# 7. SONUÇ ÖZETİ
# ============================================================================
print("\n" + "=" * 80)
print("SONUÇ ÖZETİ")
print("=" * 80)

print("\n💡 BULGULAR:")
if len(df_race) > 1:
    print(f"   🌍 Irk bazlı fairness ratio: {fairness_ratio_race:.3f}")
    if fairness_ratio_race < 0.80:
        print("      ⚠️  Model, farklı ırklara farklı tahminler yapıyor!")
    else:
        print("      ✅ Model, ırk bazında kabul edilebilir fairness gösteriyor")

if len(df_gender) > 1:
    print(f"   👤 Cinsiyet bazlı fairness ratio: {fairness_ratio_gender:.3f}")
    if fairness_ratio_gender < 0.80:
        print("      ⚠️  Model, farklı cinsiyetlere farklı tahminler yapıyor!")
    else:
        print("      ✅ Model, cinsiyet bazında kabul edilebilir fairness gösteriyor")

print("\n📌 TEZ İÇİN ÖNEMLİ NOKTALAR:")
print("   1. Sistemik bias (EDA'da tespit edildi) vs Model bias (bu analiz)")
print("   2. Model, ırk/cinsiyet faktörlerini DOĞRUDAN kullanmıyor")
print("   3. Ancak dolaylı bias (redlining, sosyoekonomik) olabilir")
print("   4. Fairness-aware ML gelecek çalışmalarda uygulanabilir")

print("\n" + "=" * 80)
print(f"✅ ANALİZ TAMAMLANDI! Çıktılar: {output_dir}/")
print("=" * 80)
print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
