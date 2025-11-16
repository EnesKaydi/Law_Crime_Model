import pandas as pd
import numpy as np

df = pd.read_csv('wcld_Processed_For_Model.csv')
jail = df[df['jail'] > 0]['jail']

print('AYKIRI VERİ (OUTLIER) ANALİZİ')
print('=' * 60)
print(f'Min: {jail.min():.2f} gün')
print(f'Max: {jail.max():.2f} gün = {jail.max()/365:.1f} yıl')
print(f'Ortalama: {jail.mean():.1f} gün')
print(f'Medyan: {jail.median():.0f} gün')
print(f'Std Sapma: {jail.std():.1f}')

Q1 = jail.quantile(0.25)
Q3 = jail.quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
outliers = jail[jail > upper]

print(f'\nIQR Yöntemi ile Outlier:')
print(f'Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}')
print(f'Outlier eşiği: {upper:.0f} gün üzeri')
print(f'Outlier sayısı: {len(outliers):,} ({len(outliers)/len(jail)*100:.1f}%)')

extreme = jail[jail > 3650]
print(f'\nExtreme Outliers (10+ yıl):')
print(f'Sayı: {len(extreme):,} ({len(extreme)/len(jail)*100:.3f}%)')

p99 = jail.quantile(0.99)
above_99 = jail[jail > p99]
print(f'\n99 Percentile üzeri ({p99:.0f} gün):')
print(f'Sayı: {len(above_99):,} ({len(above_99)/len(jail)*100:.1f}%)')
