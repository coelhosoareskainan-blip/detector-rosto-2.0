import streamlit as st
import numpy as np
from PIL import Image
import face_recognition
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av
import os

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Reconhecimento Facial 2.0", layout="centered")

# CSS para limpar a interface
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🕵️ Detector de Rosto 2.0")

DB_FILE = "face_db.npz"
LIMIAR = 0.5

# BANCO DE DADOS
def load_db():
    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE, allow_pickle=True)
        return {key: data[key] for key in data}
    return {}

def save_db(db):
    np.savez(DB_FILE, **db)

if 'db' not in st.session_state:
    st.session_state.db = load_db()

# IA
def processar_imagem(image_np):
    if image_np.shape[2] == 4: 
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    locais = face_recognition.face_locations(image_np)
    encodings = face_recognition.face_encodings(image_np, locais)
    return encodings, locais

def identificar_pessoa(encoding_desconhecido, db):
    if not db: return "Desconhecido", 1.0
    known_encodings = list(db.values())
    known_names = list(db.keys())
    dists = face_recognition.face_distance(known_encodings, encoding_desconhecido)
    nome = "Desconhecido"
    min_dist = 1.0
    if len(dists) > 0:
        min_dist = min(dists)
        if min_dist < LIMIAR:
            idx_min = np.argmin(dists)
            nome = known_names[idx_min]
    return nome, min_dist

# INTERFACE
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Câmera", "📤 Upload", "➕ Cadastro", "⚙️ Admin"])

with tab1:
    st.header("Monitoramento Real")
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    class VideoProcessor(VideoTransformerBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            locais = face_recognition.face_locations(img_rgb)
            encodings = face_recognition.face_encodings(img_rgb, locais)
            for (top, right, bottom, left), face_encoding in zip(locais, encodings):
                nome, dist = identificar_pessoa(face_encoding, st.session_state.db)
                color = (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255)
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                cv2.putText(img, f"{nome} ({dist:.2f})", (left, bottom+20), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    webrtc_streamer(key="streamer", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

with tab2:
    st.header("Verificar Foto")
    uploaded = st.file_uploader("Envie foto", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        image = np.array(Image.open(uploaded).convert('RGB'))
        st.image(image, width=300)
        encodings, _ = processar_imagem(image)
        if not encodings: st.warning("Sem rosto.")
        else:
            for enc in encodings:
                nome, dist = identificar_pessoa(enc, st.session_state.db)
                st.write(f"Resultado: **{nome}** ({dist:.2f})")

with tab3:
    st.header("Novo Cadastro")
    nome = st.text_input("Nome")
    foto = st.file_uploader("Foto única", key="cad")
    if st.button("Salvar") and nome and foto:
        img = np.array(Image.open(foto).convert('RGB'))
        encs, _ = processar_imagem(img)
        if len(encs) == 1:
            st.session_state.db[nome] = encs[0]
            save_db(st.session_state.db)
            st.success("Salvo!")
        else: st.error("A foto deve ter exatamente 1 rosto.")

with tab4:
    st.write(f"Cadastros: {len(st.session_state.db)}")
    for n in list(st.session_state.db.keys()):
        c1, c2 = st.columns([0.8, 0.2])
        c1.write(n)
        if c2.button("🗑️", key=n):
            del st.session_state.db[n]
            save_db(st.session_state.db)
            st.rerun()
