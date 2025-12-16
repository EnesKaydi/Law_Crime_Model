# 4.2. GELİŞTİRİLEN HİBRİT MODEL MİMARİSİ

Bu çalışmada, tek bir model yerine, davaları niteliklerine göre ayıran ve uzmanlaşmış alt modellere yönlendiren "Hibrit Uzmanlar Mimarisi" (Mixture of Experts) kullanılmıştır.

## 4.2.1. Yönlendirici (Router) Algoritması
Sistemin giriş kapısı olan Router, gelen davanın "Hafif/Orta" (Mainstream) mi yoksa "Ağır/Nadir" (High Severity) mi olduğuna karar verir.

### 🔄 Router Performansı (Confusion Matrix)
Aşağıdaki karmaşıklık matrisi (Confusion Matrix), Router modelinin davaları ne kadar doğru yönlendirdiğini göstermektedir.

![router_classifier_confusion_matrix](router_classifier_confusion_matrix.png)
*Şekil: Router Classifier Confusion Matrix*

## 4.2.2. Mimarinin Avantajları
*   **Uzmanlaşma:** Hafif suçlar için eğitilen model, hırsızlık gibi sık görülen suçlarda uzmanlaşırken; ağır suçlar modeli cinayet veya cinsel saldırı gibi nadir ama kritik vakalara odaklanmıştır.
*   **Başarı:** Router'ın %89 üzerindeki doğru yönlendirme başarısı, hibrit yapının temelini sağlamlaştırmıştır.

---
**Ek Dosyalar:**
_Ek dosya yok._
