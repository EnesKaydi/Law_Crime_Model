"""
📊 OVERALL SYSTEM PERFORMANCE CALCULATOR
=========================================

Ağırlıklı R² hesaplama:
- Mainstream Model: 70.43% R² (92.5% vaka)
- High Severity Model: 60.53% R² (7.5% vaka)
- Router Accuracy: 89.33%

Genel sistem performansını hesapla.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import joblib
from pathlib import Path

# Paths
VERI_YOLU = "/Users/muhammedeneskaydi/PycharmProjects/LAW/wcld.csv"
MODEL_DIR = Path("../model_data_v2_interactions")

THRESHOLD = 3000
RANDOM_STATE = 42

def calculate_overall_performance():
    """Genel sistem performansını hesapla"""
    print("="*70)
    print("📊 OVERALL SYSTEM PERFORMANCE CALCULATION")
    print("="*70)
    
    # 1. Veri yükle
    print(f"\n📂 Veri yükleniyor: {VERI_YOLU}")
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    
    # Filtreleme
    df = df[df['jail'] > 300].copy()
    ust_sinir = df['jail'].quantile(0.995)
    df = df[df['jail'] <= ust_sinir].copy()
    
    print(f"✅ Toplam vaka: {len(df):,}")
    
    # Segmentlere ayır
    df_mainstream = df[df['jail'] <= THRESHOLD].copy()
    df_high = df[df['jail'] > THRESHOLD].copy()
    
    mainstream_pct = len(df_mainstream) / len(df) * 100
    high_pct = len(df_high) / len(df) * 100
    
    print(f"\n📊 Veri Dağılımı:")
    print(f"   • Mainstream (≤3000 gün): {len(df_mainstream):,} vaka ({mainstream_pct:.1f}%)")
    print(f"   • High Severity (>3000 gün): {len(df_high):,} vaka ({high_pct:.1f}%)")
    
    # 2. Model performansları (bilinen değerler)
    print(f"\n📈 Model Performansları:")
    
    mainstream_r2 = 0.7043
    high_r2_old = 0.3337
    high_r2_new = 0.6053
    router_acc = 0.8933
    
    print(f"   • Mainstream Model R²: {mainstream_r2:.4f} ({mainstream_r2*100:.2f}%)")
    print(f"   • High Severity Model R² (OLD): {high_r2_old:.4f} ({high_r2_old*100:.2f}%)")
    print(f"   • High Severity Model R² (NEW): {high_r2_new:.4f} ({high_r2_new*100:.2f}%)")
    print(f"   • Router Accuracy: {router_acc:.4f} ({router_acc*100:.2f}%)")
    
    # 3. Ağırlıklı R² hesaplama
    print(f"\n🧮 AĞIRLIKLI R² HESAPLAMA:")
    print("="*70)
    
    # Eski sistem
    weighted_r2_old = (mainstream_pct/100 * mainstream_r2) + (high_pct/100 * high_r2_old)
    
    # Yeni sistem
    weighted_r2_new = (mainstream_pct/100 * mainstream_r2) + (high_pct/100 * high_r2_new)
    
    print(f"\n📊 ESKİ SİSTEM (High Severity: 33.37%):")
    print(f"   = ({mainstream_pct:.1f}% × {mainstream_r2:.4f}) + ({high_pct:.1f}% × {high_r2_old:.4f})")
    print(f"   = ({mainstream_pct/100:.3f} × {mainstream_r2:.4f}) + ({high_pct/100:.3f} × {high_r2_old:.4f})")
    print(f"   = {mainstream_pct/100 * mainstream_r2:.4f} + {high_pct/100 * high_r2_old:.4f}")
    print(f"   = {weighted_r2_old:.4f} ({weighted_r2_old*100:.2f}%)")
    
    print(f"\n📊 YENİ SİSTEM (High Severity: 60.53%):")
    print(f"   = ({mainstream_pct:.1f}% × {mainstream_r2:.4f}) + ({high_pct:.1f}% × {high_r2_new:.4f})")
    print(f"   = ({mainstream_pct/100:.3f} × {mainstream_r2:.4f}) + ({high_pct/100:.3f} × {high_r2_new:.4f})")
    print(f"   = {mainstream_pct/100 * mainstream_r2:.4f} + {high_pct/100 * high_r2_new:.4f}")
    print(f"   = {weighted_r2_new:.4f} ({weighted_r2_new*100:.2f}%)")
    
    # İyileşme
    improvement = weighted_r2_new - weighted_r2_old
    improvement_pct = (improvement / weighted_r2_old) * 100
    
    print(f"\n🚀 İYİLEŞME:")
    print(f"   • Eski: {weighted_r2_old:.4f} ({weighted_r2_old*100:.2f}%)")
    print(f"   • Yeni: {weighted_r2_new:.4f} ({weighted_r2_new*100:.2f}%)")
    print(f"   • Fark: +{improvement:.4f} (+{improvement*100:.2f} puan)")
    print(f"   • İyileşme: +{improvement_pct:.2f}%")
    
    # Router etkisi
    print(f"\n🎯 ROUTER ETKİSİ:")
    print(f"   Router Accuracy: {router_acc*100:.2f}%")
    print(f"   → Vakaların %{router_acc*100:.2f}'i doğru modele yönlendiriliyor")
    print(f"   → Yanlış yönlendirme: %{(1-router_acc)*100:.2f}")
    
    # Final sistem performansı (Router dahil)
    # Basitleştirilmiş: Router doğru yönlendirdiğinde model performansı, yanlış yönlendirdiğinde düşük performans
    # Gerçek hesaplama daha karmaşık ama yaklaşık olarak:
    effective_r2 = router_acc * weighted_r2_new
    
    print(f"\n🏆 FİNAL SİSTEM PERFORMANSI (Router Dahil):")
    print(f"   Effective R² ≈ {effective_r2:.4f} ({effective_r2*100:.2f}%)")
    print(f"   (Router Accuracy × Weighted R²)")
    
    # Özet tablo
    print(f"\n{'='*70}")
    print(f"📊 ÖZET TABLO")
    print(f"{'='*70}")
    print(f"\n{'Metrik':<40} {'Eski':<15} {'Yeni':<15} {'İyileşme':<15}")
    print(f"{'-'*70}")
    print(f"{'Mainstream R²':<40} {mainstream_r2:.4f}         {mainstream_r2:.4f}         -")
    print(f"{'High Severity R²':<40} {high_r2_old:.4f}         {high_r2_new:.4f}         +{(high_r2_new-high_r2_old):.4f}")
    print(f"{'Ağırlıklı R² (Log Scale)':<40} {weighted_r2_old:.4f}         {weighted_r2_new:.4f}         +{improvement:.4f}")
    print(f"{'Effective R² (Router Dahil)':<40} {router_acc*weighted_r2_old:.4f}         {effective_r2:.4f}         +{effective_r2 - router_acc*weighted_r2_old:.4f}")
    print(f"{'-'*70}")
    
    print(f"\n✅ HESAPLAMA TAMAMLANDI!")
    
    return {
        'mainstream_r2': mainstream_r2,
        'high_r2_old': high_r2_old,
        'high_r2_new': high_r2_new,
        'weighted_r2_old': weighted_r2_old,
        'weighted_r2_new': weighted_r2_new,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'router_acc': router_acc,
        'effective_r2': effective_r2,
        'mainstream_pct': mainstream_pct,
        'high_pct': high_pct
    }

if __name__ == "__main__":
    results = calculate_overall_performance()
