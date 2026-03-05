
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
OUTPUT_DIR = Path("../outputs/interaction_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_interactions():
    print(f"📂 Veri yükleniyor: {VERI_YOLU}")
    try:
        df = pd.read_csv(VERI_YOLU, low_memory=False)
    except FileNotFoundError:
        print("❌ HATA: Dosya bulunamadı!")
        return

    # Filtreleme (Ana kitle üzerinde analiz yapalım)
    if 'jail' not in df.columns: return
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    # Log Target
    df['log_jail'] = np.log1p(df['jail'])
    
    print(f"✅ Analiz Verisi: {len(df)} satır")

    # 1. SAYISAL ETKİLEŞİMLER (Numerical Interactions)
    # Mantıksal olarak çarpıldığında anlam ifade edebilecek kolonlar
    num_cols = [
        'highest_severity', 'is_recid_new', 'age_offense', 
        'prior_felony', 'prior_misdemeanor', 'avg_hist_jail',
        'age_judge', 'violent_crime'
    ]
    
    # Eksikleri doldur
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mean())

    print("\n🔍 1. Sayısal Etkileşimler Taranıyor (Çarpım/Bölüm)...")
    results = []
    
    # İkili kombinasyonlar
    for col1, col2 in combinations(num_cols, 2):
        if col1 not in df.columns or col2 not in df.columns: continue
        
        # Çarpım (Interaction)
        new_col_name = f"{col1}_x_{col2}"
        new_col_val = df[col1] * df[col2]
        corr = new_col_val.corr(df['log_jail'])
        results.append({'feature': new_col_name, 'corr': abs(corr), 'type': 'multiplication'})
        
        # Fark (Difference) - Örn: Hakim yaşı - Suçlu yaşı
        if 'age' in col1 and 'age' in col2:
            new_col_name_diff = f"{col1}_minus_{col2}"
            new_col_val_diff = df[col1] - df[col2]
            corr_diff = new_col_val_diff.corr(df['log_jail'])
            results.append({'feature': new_col_name_diff, 'corr': abs(corr_diff), 'type': 'difference'})

    # Sonuçları Sırala
    res_df = pd.DataFrame(results).sort_values(by='corr', ascending=False)
    print("\n🏆 En Yüksek Korelasyonlu Yeni Adaylar (Top 10):")
    print(res_df.head(10))
    
    # Orijinal korelasyonlarla kıyaslayalım
    print("\n(Referans) Orijinal Özelliklerin Korelasyonları:")
    orig_corrs = []
    for c in num_cols:
        if c in df.columns:
            orig_corrs.append({'feature': c, 'corr': abs(df[c].corr(df['log_jail']))})
    print(pd.DataFrame(orig_corrs).sort_values(by='corr', ascending=False).head(5))

    # 2. HAKİM x SUÇ ŞİDDETİ ANALİZİ (Categorical x Numerical)
    # Bazı hakimler ağır suçlarda daha mı acımasız? (Eğim farkı var mı?)
    print("\n🔍 2. Hakim x Suç Şiddeti Analizi...")
    
    # En çok davası olan 20 hakim
    top_judges = df['judge_id'].value_counts().head(10).index.tolist()
    
    plt.figure(figsize=(12, 6))
    for judge in top_judges:
        subset = df[df['judge_id'] == judge]
        # Her hakimin Severity vs Jail eğilimi
        sns.regplot(x='highest_severity', y='log_jail', data=subset, scatter=False, label=f'Judge {judge}', ci=None)
    
    plt.title('Hakimlerin Suç Şiddetine Göre Ceza Eğilimleri (Eğim Farklılıkları)')
    plt.xlabel('Suç Şiddeti (Highest Severity)')
    plt.ylabel('Ceza (Log Days)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "judge_severity_interaction.png")
    print("   -> Grafik kaydedildi: judge_severity_interaction.png")
    
    # 3. AGE GAP (Suçlu Yaşı vs Hakim Yaşı)
    # Genç hakim yaşlı suçluya, veya tam tersi duruma nasıl bakıyor?
    if 'age_judge' in df.columns and 'age_offense' in df.columns:
        df['age_gap'] = df['age_judge'] - df['age_offense']
        corr_gap = df['age_gap'].corr(df['log_jail'])
        print(f"\n👴⚖️ Hakim-Suçlu Yaş Farkı (Age Gap) Korelasyonu: {corr_gap:.4f}")
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='age_gap', y='log_jail', data=df.sample(5000), alpha=0.3)
        plt.title('Hakim-Suçlu Yaş Farkı vs Ceza')
        plt.xlabel('Yaş Farkı (Hakim - Suçlu)')
        plt.ylabel('Ceza (Log)')
        plt.savefig(OUTPUT_DIR / "age_gap_analysis.png")
        
    # 4. ŞİDDET VE CİNSİYET (Violent x Sex)
    if 'violent_crime' in df.columns and 'sex' in df.columns:
        print("\n👫👊 Cinsiyet ve Şiddet Suçu Etkileşimi:")
        # Erkek+Şiddet vs Kadın+Şiddet ortalamaları
        summary = df.groupby(['sex', 'violent_crime'])['jail'].mean().reset_index()
        print(summary)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='sex', y='jail', hue='violent_crime', data=df)
        plt.title('Cinsiyet ve Şiddet Suçu Etkileşimi')
        plt.ylabel('Ortalama Ceza (Gün)')
        plt.savefig(OUTPUT_DIR / "sex_violent_interaction.png")

    print(f"\n💾 Tüm analiz grafikleri kaydedildi: {OUTPUT_DIR}")
    
    # ÖNERİ
    best_interaction = res_df.iloc[0]
    print(f"\n💡 SONUÇ: En umut verici yeni özellik: '{best_interaction['feature']}' (Corr: {best_interaction['corr']:.4f})")
    print("Bu özelliği ve 'Age Gap' özelliğini modele eklemeyi deneyebiliriz.")

if __name__ == "__main__":
    analyze_interactions()
