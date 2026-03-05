# 4.3. ARAŞTIRMA BULGULARI VE PERFORMANS ANALİZİ

Geliştirilen yapay zeka modelinin tahmin başarısı, bilimsel metrikler ve hata analizleri ile bu bölümde sunulmuştur.

## 4.3.1. Genel Model Performansı
Model, test veri seti üzerinde **%83.65 R²** skoruna ulaşarak, yargı kararlarındaki varyansın büyük kısmını açıklamayı başarmıştır.

### 📈 Bilimsel Analiz Grafikleri
Aşağıdaki grafikler, modelin tahminleri ile gerçek değerler arasındaki ilişkiyi ve hataların dağılımını göstermektedir.

![high_severity_analysis_01_distribution_analysis](high_severity_analysis_01_distribution_analysis.png)
*Şekil: High Severity Analysis 01 Distribution Analysis*

![high_severity_analysis_02_feature_importance_comparison](high_severity_analysis_02_feature_importance_comparison.png)
*Şekil: High Severity Analysis 02 Feature Importance Comparison*

![high_severity_analysis_03_error_patterns](high_severity_analysis_03_error_patterns.png)
*Şekil: High Severity Analysis 03 Error Patterns*

![high_severity_analysis_04_improvement_comparison](high_severity_analysis_04_improvement_comparison.png)
*Şekil: High Severity Analysis 04 Improvement Comparison*

![performance_hata_dagilim_analizi](performance_hata_dagilim_analizi.png)
*Şekil: Performance Hata Dagilim Analizi*

![performance_kategori_bazli_performans](performance_kategori_bazli_performans.png)
*Şekil: Performance Kategori Bazli Performans*

![scientific_analysis_01_variance_decomposition](scientific_analysis_01_variance_decomposition.png)
*Şekil: Scientific Analysis 01 Variance Decomposition*

![scientific_analysis_02_feature_correlations](scientific_analysis_02_feature_correlations.png)
*Şekil: Scientific Analysis 02 Feature Correlations*

![scientific_analysis_03_error_categorization](scientific_analysis_03_error_categorization.png)
*Şekil: Scientific Analysis 03 Error Categorization*

![scientific_analysis_04_theoretical_limit](scientific_analysis_04_theoretical_limit.png)
*Şekil: Scientific Analysis 04 Theoretical Limit*

## 4.3.2. Ağır Suçlarda (High Severity) İyileştirme
Tez çalışmasının en önemli katkılarından biri, tahmin edilmesi zor olan ağır suçlardaki başarı artışıdır.
*   **Eski Başarı:** %33.37
*   **Yeni Başarı:** %60.53
*   **İyileşme:** +%81.4

Bu iyileşme, `high_severity_analysis_improvement_comparison.png` grafiğinde net bir şekilde görülmektedir.

---
**Ek Dosyalar:**
- [high_severity_analysis_diagnostic_report.md](high_severity_analysis_diagnostic_report.md)
- [high_severity_analysis_improvement_results.md](high_severity_analysis_improvement_results.md)
- [performance_en_iyi_tahminler.csv](performance_en_iyi_tahminler.csv)
- [performance_en_kotu_tahminler.csv](performance_en_kotu_tahminler.csv)
- [performance_kategori_metrikleri.csv](performance_kategori_metrikleri.csv)
- [scientific_analysis_scientific_analysis_report.md](scientific_analysis_scientific_analysis_report.md)
