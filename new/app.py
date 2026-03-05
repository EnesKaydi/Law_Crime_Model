import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predictor import TextPredictor

st.set_page_config(
    page_title="AI-LAW | Hukuk Asistanı",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Ana arka plan */
    .stApp {
        background-color: #1a1a2e;
    }
    
    /* Başlık rengi */
    h1, h2, h3, h4 {
        color: #eaeaea !important;
    }
    
    /* Input alanı */
    .stTextArea textarea {
        background-color: #16213e;
        color: #eaeaea;
        border: 1px solid #0f3460;
    }
    
    /* Buton */
    .stButton > button {
        background-color: #0f3460;
        color: #eaeaea;
        border: 1px solid #e94560;
    }
    
    /* Sonuç kutuları */
    .result-box {
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #0f3460;
        margin: 10px 0;
    }
    
    /* Bilgi metinleri */
    p, li, div {
        color: #b8b8b8 !important;
    }
    
    /* Success/Error mesajları */
    .stSuccess, .stError, .stWarning, .stInfo {
        background-color: #16213e;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #16213e;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ AI-LAW Hukuk Asistanı")
st.markdown("### Metin Tabanlı Ceza Tahmin Sistemi")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Olay Metni Girin")
    
    metin = st.text_area(
        "Olayı detaylı şekilde yazın:",
        height=200,
        placeholder="Örnek: Adamın biri markete girip 5000 TL değerinde ürün çaldı..."
    )
    
    tahmin_buton = st.button("🔍 Tahmin Et")

with col2:
    st.subheader("📋 Sonuç")
    
    if tahmin_buton and metin:
        with st.spinner("Analiz ediliyor..."):
            predictor = TextPredictor()
            sonuc = predictor.predict(metin)
            
            if sonuc["success"]:
                data = sonuc["data"]
                
                st.markdown(f"""
                <div class="result-box">
                    <h4 style="color: #4fc3f7 !important; margin-bottom: 10px;">📌 Suç Bilgileri</h4>
                    <p><strong>Suç Kategorisi:</strong> {data['suç_kategorisi']} ({data['suç_kategorisi_en']})</p>
                    <p><strong>TCK Madde:</strong> {data['tck_madde']}</p>
                    <p><strong>Açıklama:</strong> {data['tck_aciklama']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="result-box">
                    <h4 style="color: #81c784 !important; margin-bottom: 10px;">⚖️ Ceza Tahmini</h4>
                    <p><strong>Ceza Aralığı:</strong> {data['ceza_araligi']}</p>
                    <p><strong>Gün Aralığı:</strong> {data['ceza_araligi_gun']} gün</p>
                    <hr style="border-color: #0f3460;">
                    <p style="font-size: 18px;"><strong>Tahmini Ceza:</strong> {data['model_tahmini_gun']} gün ({data['model_tahmini_yil']} yıl)</p>
                    <p><strong>Güven Oranı:</strong> {data['guven_orani']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔧 Çıkarılan Özellikler"):
                    st.json(data["ozellikler"])
            
            else:
                st.error(f"Hata: {sonuc['hata']}")
                st.warning("Lütfen metni daha detaylı yazın.")
    
    elif tahmin_buton and not metin:
        st.warning("Lütfen bir metin girin!")
    
    else:
        st.info("Sol tarafa olayı yazın ve 'Tahmin Et' butonuna basın")

st.markdown("---")

st.markdown("""
### 💡 Örnek Olaylar

Birini seçip deneyebilirsiniz:

1. **Hırsızlık:** "Adam marketten 5000 TL değerinde ürün çaldı"
2. **Silahlı Soygun:** "Şahıs bankaya girerek silahla tehdit etti ve para aldı"  
3. **Yaralama:** "Kişi sokakta başka birini dövdü ve yaraladı"
4. **Uyuşturucu:** "Polis, şahsın evinde uyuşturucu buldu"
5. **Silah Bulundurma:** "Şüphelinin üzerinde yasadışı silah bulundu"
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>AI-LAW | Yapay Zeka Destekli Hukuk Asistanı</p>
    <p>Bu sistem tahmin amaçlıdır. Gerçek kararlar için avukatınıza danışın.</p>
</div>
""", unsafe_allow_html=True)
