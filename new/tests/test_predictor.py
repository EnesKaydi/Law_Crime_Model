import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from predictor import TextPredictor


class TestTextPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Predictor'ı bir kez başlat (tüm testlerde paylaş)"""
        cls.predictor = TextPredictor()

    # ── SUÇA GÖRE TESPİT TESTLERİ ─────────────────────────────────────────────

    def test_theft(self):
        result = self.predictor.predict("Adam marketten 5000 TL değerinde ürün çaldı")
        self.assertTrue(result["success"], "Hırsızlık tespiti başarısız")
        self.assertIn("hırsızlık", result["data"]["suç_kategorisi"].lower())

    def test_armed_robbery(self):
        result = self.predictor.predict("Şahıs bankaya girerek silahla tehdit etti ve zorla para aldı")
        self.assertTrue(result["success"], "Silahlı soygun tespiti başarısız")
        self.assertIn("silahlı soygun", result["data"]["suç_kategorisi"].lower())

    def test_robbery_priority_over_theft(self):
        """Soygun, hırsızlıktan yüksek önceliğe sahip olmalı"""
        result = self.predictor.predict("Kişi silahla soygun yaptı ve para aldı")
        self.assertTrue(result["success"])
        kategorisi = result["data"]["suç_kategorisi"].lower()
        self.assertNotEqual(kategorisi, "hırsızlık",
                            "Soygun, hırsızlık olarak sınıflandırılmamalı")

    def test_drug_possession(self):
        result = self.predictor.predict("Polis şüphelinin evinde uyuşturucu buldu")
        self.assertTrue(result["success"], "Uyuşturucu bulundurma tespiti başarısız")
        self.assertIn("uyuşturucu", result["data"]["suç_kategorisi"].lower())

    def test_murder(self):
        result = self.predictor.predict("Zanlı komşusunu kasten öldürdü")
        self.assertTrue(result["success"], "Kasten öldürme tespiti başarısız")
        self.assertIn("öldürme", result["data"]["suç_kategorisi"].lower())

    def test_battery(self):
        result = self.predictor.predict("Kişi sokakta başka birini dövdü ve yaraladı")
        self.assertTrue(result["success"], "Yaralama tespiti başarısız")
        self.assertIn("yaralama", result["data"]["suç_kategorisi"].lower())

    def test_weapon_possession(self):
        result = self.predictor.predict("Şüphelinin üzerinde ruhsatsız ve yasadışı silah bulundu")
        self.assertTrue(result["success"], "Silah bulundurma tespiti başarısız")
        self.assertIn("silah bulundurma", result["data"]["suç_kategorisi"].lower())

    def test_drunk_driving(self):
        result = self.predictor.predict("Sürücü alkollü araç kullandı ve trafik kazasına neden oldu")
        self.assertTrue(result["success"], "Alkollü araç kullanma tespiti başarısız")
        self.assertIn("alkollü", result["data"]["suç_kategorisi"].lower())

    # ── HATA DURUMU TESTLERİ ──────────────────────────────────────────────────

    def test_empty_text(self):
        result = self.predictor.predict("")
        self.assertFalse(result["success"], "Boş metin başarısız dönmeli")

    def test_unknown_crime(self):
        result = self.predictor.predict("Bugün hava çok güzeldi, pikniğe gittik")
        self.assertFalse(result["success"], "Tanımsız suç başarısız dönmeli")

    # ── ÇIKTI YAPISI TESTLERİ ─────────────────────────────────────────────────

    def test_output_structure(self):
        """Başarılı tahmin gerekli alanları içermeli"""
        result = self.predictor.predict("Adam marketten ürün çaldı")
        self.assertTrue(result["success"])
        data = result["data"]
        zorunlu_alanlar = [
            "suç_kategorisi", "suç_kategorisi_en", "tck_madde",
            "tck_aciklama", "ceza_araligi", "ceza_araligi_gun",
            "model_tahmini_gun", "model_tahmini_yil", "guven_orani", "kaynak"
        ]
        for alan in zorunlu_alanlar:
            self.assertIn(alan, data, f"'{alan}' alanı eksik")

    def test_prediction_in_tck_range(self):
        """Model tahmini TCK aralığında olmalı"""
        result = self.predictor.predict("Adam marketten ürün çaldı")
        self.assertTrue(result["success"])
        data = result["data"]
        gun = data["model_tahmini_gun"]
        # hırsızlık: 730-1825 gün
        self.assertGreaterEqual(gun, 730, "Tahmin minimum ceza gününün altında")
        self.assertLessEqual(gun, 1825, "Tahmin maksimum ceza gününün üstünde")


if __name__ == "__main__":
    unittest.main(verbosity=2)
