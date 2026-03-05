
import os
from pathlib import Path

BASE_DIR = Path("final_sonuclar_cıktılar")

# Bölüm İçerik Şablonları
TEMPLATES = {
    "4.1_Veri_Analizi": """# 4.1. VERİ SETİ ANALİZİ VE ÖN İŞLEME

Bu bölümde, Wisconsin Eyaleti mahkeme kayıtlarından elde edilen veri setinin yapısal özellikleri, değişkenler arası ilişkiler ve veri temizleme süreçleri detaylandırılmıştır.

## 4.1.1. Veri Setinin Genel Yapısı
Veri seti, suçun niteliği, sanığın demografik özellikleri ve geçmiş suç kayıtları gibi 50'den fazla değişken içermektedir. Modelin başarısını artırmak adına, veri setinde bulunan gürültülü veriler ve aykırı değerler (outliers) temizlenmiştir.

### 📊 Korelasyon ve İlişki Analizi
Aşağıdaki analizler, veri setindeki değişkenlerin birbirleriyle olan ilişkisini göstermektedir. Özellikle "Judge" (Hakim) değişkeninin ceza süreleri üzerindeki etkisi incelenmiştir.

{images}

## 4.1.2. Bulguların Yorumlanması
*   **Hakim Etkisi:** `01_detayli_analiz_judge_analizi.png` grafiğinde görüldüğü üzere, farklı hakimlerin benzer davalarda verdikleri ceza süreleri arasında belirgin farklar bulunmaktadır. Bu durum, modelin "Hakim ID" bilgisini bir özellik (feature) olarak kullanmasının gerekliliğini ortaya koymuştur.
*   **Değişken İlişkileri:** Korelasyon matrisi, suçun ciddiyeti (`severity`) ile ceza süresi arasında güçlü bir pozitif ilişki olduğunu doğrulamaktadır.

---
**Ek Dosyalar:**
{files}
""",

    "4.2_Model_Mimarisi": """# 4.2. GELİŞTİRİLEN HİBRİT MODEL MİMARİSİ

Bu çalışmada, tek bir model yerine, davaları niteliklerine göre ayıran ve uzmanlaşmış alt modellere yönlendiren "Hibrit Uzmanlar Mimarisi" (Mixture of Experts) kullanılmıştır.

## 4.2.1. Yönlendirici (Router) Algoritması
Sistemin giriş kapısı olan Router, gelen davanın "Hafif/Orta" (Mainstream) mi yoksa "Ağır/Nadir" (High Severity) mi olduğuna karar verir.

### 🔄 Router Performansı (Confusion Matrix)
Aşağıdaki karmaşıklık matrisi (Confusion Matrix), Router modelinin davaları ne kadar doğru yönlendirdiğini göstermektedir.

{images}

## 4.2.2. Mimarinin Avantajları
*   **Uzmanlaşma:** Hafif suçlar için eğitilen model, hırsızlık gibi sık görülen suçlarda uzmanlaşırken; ağır suçlar modeli cinayet veya cinsel saldırı gibi nadir ama kritik vakalara odaklanmıştır.
*   **Başarı:** Router'ın %89 üzerindeki doğru yönlendirme başarısı, hibrit yapının temelini sağlamlaştırmıştır.

---
**Ek Dosyalar:**
{files}
""",

    "4.3_Performans_Bulgulari": """# 4.3. ARAŞTIRMA BULGULARI VE PERFORMANS ANALİZİ

Geliştirilen yapay zeka modelinin tahmin başarısı, bilimsel metrikler ve hata analizleri ile bu bölümde sunulmuştur.

## 4.3.1. Genel Model Performansı
Model, test veri seti üzerinde **%83.65 R²** skoruna ulaşarak, yargı kararlarındaki varyansın büyük kısmını açıklamayı başarmıştır.

### 📈 Bilimsel Analiz Grafikleri
Aşağıdaki grafikler, modelin tahminleri ile gerçek değerler arasındaki ilişkiyi ve hataların dağılımını göstermektedir.

{images}

## 4.3.2. Ağır Suçlarda (High Severity) İyileştirme
Tez çalışmasının en önemli katkılarından biri, tahmin edilmesi zor olan ağır suçlardaki başarı artışıdır.
*   **Eski Başarı:** %33.37
*   **Yeni Başarı:** %60.53
*   **İyileşme:** +%81.4

Bu iyileşme, `high_severity_analysis_improvement_comparison.png` grafiğinde net bir şekilde görülmektedir.

---
**Ek Dosyalar:**
{files}
""",

    "4.4_Aciklanabilirlik": """# 4.4. MODELİN AÇIKLANABİLİRLİĞİ (XAI)

Yapay zeka modelinin "kara kutu" olmaktan çıkarılması ve kararlarının hukuki dayanaklarının anlaşılması amacıyla SHAP (SHapley Additive exPlanations) analizi uygulanmıştır.

## 4.4.1. Özellik Önem Düzeyleri (Feature Importance)
Modelin karar verirken hangi faktörlere ne kadar ağırlık verdiği aşağıda gösterilmiştir.

### 💡 SHAP ve Etkileşim Analizleri
Bu grafikler, modelin "neden bu cezayı verdiğini" görselleştirir.

{images}

## 4.4.2. Kritik Bulgular
*   **Violent Recidivism:** `shap_analysis_shap_summary.png` grafiğinde en üstte yer alan `violent_recid` özelliği, modelin şiddet içeren mükerrer suçlara çok yüksek ceza öngördüğünü kanıtlamaktadır.
*   **Etkileşimler:** `interaction_analysis` grafikleri, yaş farkı veya cinsiyet ile şiddet suçu arasındaki karmaşık ilişkilerin model tarafından öğrenildiğini gösterir.

---
**Ek Dosyalar:**
{files}
""",

    "4.5_Etik_ve_Adalet": """# 4.5. ETİK ANALİZ VE ADALET (FAIRNESS)

Yapay zeka sistemlerinin yargı süreçlerinde kullanımı, "önyargı" (bias) riskini beraberinde getirir. Bu bölümde, modelin ırk, cinsiyet ve coğrafi bölge bazında adil davranıp davranmadığı incelenmiştir.

## 4.5.1. Irk ve Cinsiyet Yanlılığı (Race & Gender Bias)
Modelin farklı demografik gruplar için ürettiği ortalama hata payları analiz edilmiştir.

### ⚖️ Bias Analiz Grafikleri
Aşağıdaki görseller, modelin hassas gruplara yaklaşımını özetler.

{images}

## 4.5.2. Coğrafi Adalet (Geo-Analysis)
Wisconsin eyaletinin farklı ilçelerindeki (county) yargı sertliği incelenmiştir. `geo_analysis_geo_justice_score.png`, hangi bölgelerin daha sert veya daha yumuşak kararlar verdiğini haritalandırır.

---
**Ek Dosyalar:**
{files}
"""
}

def generate_chapter_content():
    print("🚀 Bölüm içerikleri oluşturuluyor...")
    
    for folder_name, template in TEMPLATES.items():
        folder_path = BASE_DIR / folder_name
        
        if not folder_path.exists():
            print(f"⚠️ Klasör bulunamadı: {folder_name}")
            continue
            
        # Klasördeki dosyaları tara
        images = []
        data_files = []
        
        for f in sorted(folder_path.glob("*")):
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Resim formatı: ![Dosya Adı](dosya_adi.png)
                images.append(f"![{f.stem}]({f.name})\n*Şekil: {f.stem.replace('_', ' ').title()}*")
            elif f.suffix.lower() in ['.csv', '.txt', '.md'] and f.name != "BOLUM_ICERIGI.md":
                data_files.append(f"- [{f.name}]({f.name})")
        
        # Şablonu doldur
        content = template.format(
            images="\n\n".join(images) if images else "_Bu klasörde görsel bulunamadı._",
            files="\n".join(data_files) if data_files else "_Ek dosya yok._"
        )
        
        # Dosyayı yaz
        output_file = folder_path / "BOLUM_ICERIGI.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"✅ Oluşturuldu: {folder_name}/BOLUM_ICERIGI.md")

if __name__ == "__main__":
    generate_chapter_content()
