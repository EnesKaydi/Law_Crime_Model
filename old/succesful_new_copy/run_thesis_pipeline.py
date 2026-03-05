import subprocess
import os
import sys
import time
from pathlib import Path

# Çalıştırılacak scriptlerin bulunduğu klasör
SCRIPTS_DIR = Path("succesful_new_copy")
LOG_DIR = Path("pipeline_logs")
LOG_DIR.mkdir(exist_ok=True)

# Çalıştırılacak dosyaların listesi (Sıralı)
# step_01'den step_29'a kadar olanları otomatik bulup sıralayalım
scripts = sorted([f for f in os.listdir(SCRIPTS_DIR) if f.startswith("step_") and f.endswith(".py")])

# İstenmeyen veya manuel çalıştırılması gereken dosyaları filtreleyebiliriz
# Örneğin web server başlatan bir script varsa buraya eklenmeli
EXCLUDE_LIST = [] 

scripts = [s for s in scripts if s not in EXCLUDE_LIST]

# Sadece step_01 ile step_29 arasındakileri alalım (kullanıcı isteği)
# Zaten sorted listesi 01-29 arası olacaktır ama emin olalım.
scripts = [s for s in scripts if 1 <= int(s.split('_')[1]) <= 29]

print(f"🚀 Toplam {len(scripts)} adet script çalıştırılacak.")
print("📂 Loglar 'pipeline_logs' klasörüne kaydedilecek.\n")

total_start_time = time.time()

for script in scripts:
    script_path = SCRIPTS_DIR / script
    log_file = LOG_DIR / f"{script.replace('.py', '.log')}"
    
    print(f"▶️  Çalıştırılıyor: {script} ...", end="", flush=True)
    
    start_time = time.time()
    
    with open(log_file, "w") as log:
        try:
            # Scripti çalıştır ve çıktıları log dosyasına yönlendir
            # cwd (current working directory) olarak scriptin olduğu klasörü değil, 
            # projenin ana dizinini kullanıyoruz ki path'ler bozulmasın.
            # Ancak scriptler "../" ile path veriyorsa, scriptin olduğu klasörde çalışması gerekebilir.
            # Dosyaları incelediğimde `MODEL_DIR = Path("../model_data_advanced")` gibi yapılar gördüm.
            # Bu demek oluyor ki scriptler `succesful_new_copy` klasörünün içinden çalıştırılmalı.
            
            process = subprocess.run(
                [sys.executable, script], 
                cwd=SCRIPTS_DIR,
                stdout=log, 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            duration = time.time() - start_time
            
            if process.returncode == 0:
                print(f" ✅ Tamamlandı ({duration:.2f} sn)")
            else:
                print(f" ❌ HATA! (Kod: {process.returncode})")
                print(f"    Detaylar için: {log_file}")
                # Hata durumunda devam edip etmeme kararı? 
                # Genelde pipeline bozulursa durmak iyidir ama kullanıcı "hepsini çalıştır" dedi.
                # Devam ediyoruz.
                
        except Exception as e:
            print(f" 💥 EXCEPTION: {e}")

total_duration = time.time() - total_start_time
print(f"\n🏁 Tüm işlemler tamamlandı. Toplam Süre: {total_duration:.2f} sn")
print(f"📄 Logları incelemek için: {LOG_DIR.absolute()}")
