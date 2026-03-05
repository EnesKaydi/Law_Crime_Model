# 4.4. MODELİN AÇIKLANABİLİRLİĞİ (XAI)

Yapay zeka modelinin "kara kutu" olmaktan çıkarılması ve kararlarının hukuki dayanaklarının anlaşılması amacıyla SHAP (SHapley Additive exPlanations) analizi uygulanmıştır.

## 4.4.1. Özellik Önem Düzeyleri (Feature Importance)
Modelin karar verirken hangi faktörlere ne kadar ağırlık verdiği aşağıda gösterilmiştir.

### 💡 SHAP ve Etkileşim Analizleri
Bu grafikler, modelin "neden bu cezayı verdiğini" görselleştirir.

![explainability_individual_predictions](explainability_individual_predictions.png)
*Şekil: Explainability Individual Predictions*

![explainability_partial_dependence_plots](explainability_partial_dependence_plots.png)
*Şekil: Explainability Partial Dependence Plots*

![explainability_permutation_importance](explainability_permutation_importance.png)
*Şekil: Explainability Permutation Importance*

![explainability_xgboost_feature_importance](explainability_xgboost_feature_importance.png)
*Şekil: Explainability Xgboost Feature Importance*

![interaction_analysis_age_gap_analysis](interaction_analysis_age_gap_analysis.png)
*Şekil: Interaction Analysis Age Gap Analysis*

![interaction_analysis_judge_severity_interaction](interaction_analysis_judge_severity_interaction.png)
*Şekil: Interaction Analysis Judge Severity Interaction*

![interaction_analysis_sex_violent_interaction](interaction_analysis_sex_violent_interaction.png)
*Şekil: Interaction Analysis Sex Violent Interaction*

![shap_analysis_shap_race](shap_analysis_shap_race.png)
*Şekil: Shap Analysis Shap Race*

![shap_analysis_shap_severity_interaction](shap_analysis_shap_severity_interaction.png)
*Şekil: Shap Analysis Shap Severity Interaction*

![shap_analysis_shap_summary](shap_analysis_shap_summary.png)
*Şekil: Shap Analysis Shap Summary*

## 4.4.2. Kritik Bulgular
*   **Violent Recidivism:** `shap_analysis_shap_summary.png` grafiğinde en üstte yer alan `violent_recid` özelliği, modelin şiddet içeren mükerrer suçlara çok yüksek ceza öngördüğünü kanıtlamaktadır.
*   **Etkileşimler:** `interaction_analysis` grafikleri, yaş farkı veya cinsiyet ile şiddet suçu arasındaki karmaşık ilişkilerin model tarafından öğrenildiğini gösterir.

---
**Ek Dosyalar:**
- [explainability_permutation_importance.csv](explainability_permutation_importance.csv)
- [explainability_xgboost_feature_importance.csv](explainability_xgboost_feature_importance.csv)
