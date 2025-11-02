Proje Metodolojisi: Veri Seti, Modelleme ve Teknolojik Altyapı
Bu çalışma kapsamında geliştirilen yapay zekâ destekli hukuk asistanı, karar verme süreçlerini desteklemeyi ve ceza sonrası toplumsal entegrasyonu kolaylaştırmayı amaçlamaktadır. Sistemin temelinde, ceza tahmini ve görev ataması yapabilen çift modüllü bir makine öğrenmesi mimarisi yer almaktadır.

Veri Seti Analizi ve Hazırlık Süreci
Modelin eğitiminde kullanılan veri seti, ABD'nin Wisconsin Eyaleti Ceza Mahkemeleri’ne ait, yaklaşık 1.5 milyon yargı kaydından oluşan geniş kapsamlı ve etiketlenmiş bir veri kümesidir.

Veri İçeriği Veri seti; suçlu bireylerin demografik ve sosyoekonomik özellikleri, suç türleri, önceki sabıka kayıtları, verilen cezalar ve tekrarlayan suç eğilimleri gibi çok boyutlu bilgileri içermektedir. Toplamda 54 kolon bulunan  veri setindeki değişkenler tematik olarak dört ana grupta toplanmıştır:


Demografik ve Sosyoekonomik Bilgiler: sex (Cinsiyet), race (Irk/Etnik köken), age_offense (Suçun işlendiği yaştaki birey) ve suçlunun yaşadığı mahallenin istatistiksel bilgileri (pct_black, pct_college, med_hhinc vb.).


Suç Bilgileri ve Davalar: case_type (Dava türü, örn: felony), wcisclass (Suç açıklaması, örn: uyuşturucu bulundurma) ve violent_crime (Suçun şiddet içerip içermediği).


Ceza Geçmişi ve Kararlar: prior_felony (Önceki ağır suçlar), prior_misdemeanor (Önceki hafif suçlar), jail (Hapis), probation (Şartlı tahliye) ve release (Serbest kalma durumu).


Tekrarlayan Suç (Recidivism): recid180d (180 gün içinde yeniden suç işleme) ve is_recid_new (Yeni bir suçla yeniden karşı karşıya kalma).

Veri Temizliği ve Örneklem Stratejisi 1.5 milyon satırlık ham veri seti üzerinde yapılan ön analizler sonucunda, tüm değişkenler açısından eksiksiz ve güvenilir olduğu belirlenen yaklaşık 357.452 satırlık bir "temiz" bölüm ana kaynak olarak seçilmiştir.

Modelin çeşitliliğini ve genelleme yeteneğini korumak amacıyla, geriye kalan yaklaşık 1.1 milyon eksik verili kayıttan rastgele %15 örneklem yöntemiyle yaklaşık 167.000 satır daha seçilerek veri setine dahil edilmiştir. Bu iki kümenin birleştirilmesiyle, model eğitiminde kullanılmak üzere yaklaşık 525.000 satırlık dengelenmiş ve zenginleştirilmiş bir "final veri seti" elde edilmiştir.

Modelleme Mimarisi ve Algoritma Seçimi
Veri seti, geliştirilen yapay zekâ mimarisinin iki temel modülünü beslemektedir:


Karar Destek Sistemi: Bu modül, hâkimlere yönelik karar desteği sunar. Bireyin demografik yapısı, suç türü ve sabıka geçmişi gibi faktörlere dayanarak uygun ceza türü (hapis, şartlı tahliye vb.) ve süresi hakkında önerilerde bulunur.


Ceza Sonrası İş Öneri Sistemi: Bu modül, hükümlü bireylerin topluma yeniden kazandırılmasını hedefler. Bireyin yaşı, cinsiyeti, aldığı ceza süresi ve suçun niteliğine (şiddet içerip içermediği) göre toplum yararına uygun kamu görevlerine yönlendirilmesini sağlar.

Özellik (Feature) ve Etiket (Label) Belirleme Modelin öğrenme sürecinde, veri setindeki kolonlar girdi (features) ve hedef (labels) olarak yapılandırılmıştır:


Girdi (Features) Olarak Kullanılan Kolonlar: sex, race, age_offense, case_type, wcisclass, prior_felony, prior_misdemeanor, violent_crime ve mahalle sosyodemografik oranları (pct_black, pct_college, med_hhinc vb.).

Etiket (Labels) Olarak Kullanılan Kolonlar: Modelin tahmin etmesi hedeflenen çıktılardır. Bunlar jail (Hapis cezası süresi), probation (Şartlı tahliye süresi) ve release (Serbest bırakılma durumu) kolonlarıdır.

Ceza Sınıflandırması ve İş Atama Stratejisi Modelin "İş Öneri Sistemi", tahmin edilen ceza süresi ve suçun niteliğine göre çalışır. Cezalar, ağırlık derecelerine göre üç kategoride sınıflandırılmış ve bu kategorilere uygun kamu hizmeti görevleri eşleştirilmiştir:


Hafif Cezalar (1–180 gün): Düşük riskli kabul edilen bu gruptakilere park temizliği, ağaç dikimi, kütüphane düzenleme, yaşlılara yardım gibi hafif fiziksel efor gerektiren işler önerilir.

Orta Düzey Cezalar (181–1080 gün): Daha disiplinli görevler içerir; örneğin, mezarlık temizliği, hayvan barınağı desteği, geri dönüşüm merkezinde malzeme ayrıştırma.


Ağır Cezalar (1081+ gün): Yüksek fiziksel dayanıklılık gerektiren ve genellikle daha ciddi suçlar için uygulanan görevlerdir; asfalt dökümü, kanalizasyon çalışmasına destek, inşaat sahası taşıma işleri, kar küreme gibi.

Makine Öğrenmesi Tercih Gerekçeleri (XGBoost) Projede, 50'den fazla değişken ve 500.000'den fazla örnek içeren bu karmaşık ve yüksek boyutlu veri setinde yüksek performans göstermesi, gradyan bazlı yöntemlerin tercih edilmesini zorunlu kılmıştır. Bu nedenle XGBoost algoritması temel alınmıştır.

XGBoost'un tercih edilmesinin temel gerekçeleri şunlardır:


Veri Yapısı ve Büyüklüğü: XGBoost, 500.000'den fazla örnek içeren yüksek boyutlu verilerdeki karmaşık ilişkileri öğrenmede üstün performans gösterir.


Eksik Veri ile Uyum: Algoritmanın, eksik verilerle (veri setine kasıtlı olarak eklenen %15'lik örneklem gibi) doğrudan çalışabilme yeteneği, modelin performansını ve gerçek dünya koşullarındaki sürdürülebilirliğini artırmaktadır.


Model Açıklanabilirliği (XAI): Hukuk gibi etik açıdan hassas bir alanda şeffaflık kritiktir. XGBoost'un SHAP (SHapley Additive exPlanations) analizleriyle birlikte kullanılabilmesi, modelin hangi girdiye dayanarak hangi kararı verdiğini açıklamaya imkân tanır.


Hız ve Uygulama Kolaylığı: Derin öğrenme yöntemlerine kıyasla çok daha az donanım gerektirmesi ve model eğitim sürelerinin kısa olması, akademik projelerde uygulanabilirliğini artırmaktadır.

Sistemin Teknolojik Altyapısı (Teknoloji Yığını)
Geliştirilen yapay zekâ destekli hukuk asistanı uygulaması, modern web teknolojileri ve makine öğrenmesi altyapılarının bir kombinasyonu üzerine kurulmuştur.


Frontend (Kullanıcı Arayüzü): Arayüz, sunucu taraflı render (SSR) desteği sunan Next.js framework'ü ile geliştirilmiştir. Kullanıcı etkileşimleri için React bileşenleri, API haberleşmesi için ise Axios kütüphanesi kullanılmıştır.


Backend (Sunucu ve API): Sunucu tarafı Python programlama dili ile geliştirilmiş olup, RESTful API yapısını oluşturmak için Flask mikro framework'ü tercih edilmiştir. Frontend ve backend arasındaki güvenli iletişim için flask-cors kütüphanesi kullanılmıştır.


Makine Öğrenmesi Katmanı: Veri işleme, analiz ve görselleştirme süreçlerinde Pandas, Numpy, Matplotlib ve Seaborn kütüphanelerinden yararlanılmıştır. Modelin eğitimi ve performans metriklerinin (RMSE, MAE, F1-score vb.) hesaplanması için XGBoost ve Scikit-learn (sklearn) kütüphaneleri kullanılmıştır. Eğitilen model çıktıları, Joblib kütüphanesi ile .pkl formatında kaydedilerek Flask API aracılığıyla çağrılabilir hale getirilmiştir.


Veritabanı Katmanı: Kullanıcı bilgileri, suç kayıtları ve model tahminlerinin kalıcı olarak saklanması için MySQL ilişkisel veritabanı yönetim sistemi kullanılmıştır.


Versiyon Kontrolü: Kod versiyonlarının takibi ve proje yönetimi için Git ve GitHub platformları kullanılmıştır.