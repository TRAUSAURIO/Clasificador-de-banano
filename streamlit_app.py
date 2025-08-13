import os
import io
import time
import csv
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------------
# Configuración / rutas
# ---------------------------
ROOT = Path(".")
MODEL_DIR = ROOT / "modelo"
MODEL_PATH = MODEL_DIR / "mejor_modelo.h5"
REGISTERED_FACES_DIR = ROOT / "rostros_registrados"
LOG_CSV = ROOT / "registro.csv"

IMG_H = 224
IMG_W = 224

# Thresholds (ajustables)
FACE_MATCH_THRESHOLD = 2200.0   # MSE threshold for naive face match (bajar = más estricto)
MOTION_PIXEL_THRESHOLD = 5000   # número de píxeles que cambian para considerar movimiento
YELLOW_PIXEL_THRESHOLD = 3000   # número de píxeles amarillos para considerar "banano detectado"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="🍌 Sistema: Login facial + Clasificador de banano", layout="wide")
st.title("🍌 Sistema seguro — Login facial, detección y clasificación")

# ---------------------------
# Utilidades de rostros
# ---------------------------

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_registered_faces(folder: Path):
    """
    Carga imágenes de rostros desde folder y crea 'encodings' simples:
    - Detecta la cara con Haarcascade, recorta, convierte a gris y redimensiona a 100x100
    - Guarda un vector (float32) por cada imagen con su nombre (nombre del archivo sin extensión)
    """
    encodings = []
    names = []
    folder.mkdir(parents=True, exist_ok=True)
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            # intentar usar toda la imagen si no hay cara detectada
            face_region = cv2.resize(gray, (100, 100))
        else:
            (x, y, w, h) = faces[0]
            face_region = gray[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, (100, 100))
        vec = face_region.flatten().astype(np.float32)
        encodings.append(vec)
        names.append(f.stem)
    return names, encodings

def face_encoding_from_image_pil(pil_img):
    """
    Dado PIL image (RGB), detecta cara y devuelve encoding simple.
    """
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB->BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        face_region = cv2.resize(gray, (100, 100))
    else:
        (x, y, w, h) = faces[0]
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (100, 100))
    return face_region.flatten().astype(np.float32)

def mse(a, b):
    return np.mean((a - b) ** 2)

# ---------------------------
# Cargar caras registradas (para login)
# ---------------------------
registered_names, registered_encodings = load_registered_faces(REGISTERED_FACES_DIR)
st.sidebar.markdown("**Usuarios registrados:**")
if registered_names:
    for n in registered_names:
        st.sidebar.write(f"- {n}")
else:
    st.sidebar.write("_No hay rostros registrados. Añade imágenes a 'rostros_registrados/' en tu repo._")

# ---------------------------
# Cargar (o permitir subir) modelo
# ---------------------------
@st.cache_resource
def load_model_if_exists(path):
    if path.exists():
        try:
            m = tf.keras.models.load_model(str(path))
            return m
        except Exception as e:
            st.sidebar.error(f"Error cargando modelo: {e}")
            return None
    return None

model = load_model_if_exists(MODEL_PATH)

st.sidebar.markdown("### Modelo")
if model is None:
    st.sidebar.warning("No se encontró `modelo/mejor_modelo.h5`. Puedes subirlo aquí (opcional).")
    uploaded_model = st.sidebar.file_uploader("Sube un modelo Keras (.h5)", type=["h5"])
    if uploaded_model is not None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.sidebar.success("Modelo subido. Recarga la página para cargarlo.")
else:
    st.sidebar.success("Modelo cargado ✅")

# ---------------------------
# Registro: crear CSV si no existe
# ---------------------------
if not LOG_CSV.exists():
    with open(LOG_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user", "event", "class", "confidence"])

# ---------------------------
# UI: Paso 1 - Login facial
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user"] = None

st.sidebar.markdown("---")
st.sidebar.write("Instrucciones:")
st.sidebar.write("1) Colócate frente a la cámara y pulsa 'Capturar para login'.")
st.sidebar.write("2) Si tu rostro está registrado, entrarás al panel principal.")

st.header("🔐 Login por reconocimiento facial")
col1, col2 = st.columns([2, 1])

with col1:
    captured = st.camera_input("Coloca tu rostro y pulsa aquí para capturar (login)")
    if captured is not None:
        pil_img = Image.open(captured)
        enc = face_encoding_from_image_pil(pil_img)
        # comparar con encodings registrados
        best_name = None
        best_mse = float("inf")
        for name, reg_enc in zip(registered_names, registered_encodings):
            val = mse(enc, reg_enc)
            if val < best_mse:
                best_mse = val
                best_name = name
        if best_name is not None and best_mse <= FACE_MATCH_THRESHOLD:
            st.success(f"✅ Hola {best_name} — acceso concedido (mse={best_mse:.1f})")
            st.session_state["authenticated"] = True
            st.session_state["user"] = best_name
            # registrar evento de login
            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.utcnow().isoformat(), best_name, "login", "", ""])
        else:
            st.error("❌ Rostro no reconocido. Si eres usuario autorizado, añade tu foto a 'rostros_registrados/'.")
            st.session_state["authenticated"] = False
else:
    st.info(f"Sesión iniciada como: **{st.session_state['user']}**")
    if st.button("Cerrar sesión"):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.experimental_rerun()

# ---------------------------
# Si autenticado: mostrar panel principal
# ---------------------------
if st.session_state["authenticated"]:
    st.markdown("---")
    st.header("📷 Panel principal — Cámara en vivo y clasificación")

    # Contenedores UI
    col_vid, col_info = st.columns([3, 1])
    with col_info:
        st.subheader("Último evento")
        last_event = st.empty()
        st.subheader("Controles")
        run_stream = st.checkbox("Activar cámara en vivo (stream)", value=True)
        save_snapshots = st.checkbox("Guardar snapshots (carpeta snapshots/)", value=False)
        st.write("Modelo cargado:" , "Sí" if model is not None else "No (sube .h5 en la barra lateral)")

    # Carpeta de snapshots
    SNAP_DIR = ROOT / "snapshots"
    if save_snapshots:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)

    # Variables compartidas entre transformer y UI (simple shared state)
    shared = {
        "last_status": "Sin actividad",
        "last_class": "",
        "last_conf": 0.0,
        "last_time": "",
    }

    # ---------------------------
    # VideoTransformer (webrtc)
    # ---------------------------
    class DetectorTransformer(VideoTransformerBase):
        def __init__(self):
            self.prev_gray = None
            self.model = model  # can be None
            self.user = st.session_state.get("user", "unk")
            self.frame_count = 0

        def detect_motion(self, gray):
            if self.prev_gray is None:
                self.prev_gray = gray
                return False
            diff = cv2.absdiff(self.prev_gray, gray)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_pixels = np.count_nonzero(th)
            self.prev_gray = gray
            return motion_pixels > MOTION_PIXEL_THRESHOLD

        def detect_yellow(self, frame_bgr):
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            lower = np.array([20, 100, 100])
            upper = np.array([30, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            yellow_pixels = np.count_nonzero(mask)
            return yellow_pixels > YELLOW_PIXEL_THRESHOLD, mask

        def classify_frame(self, frame_bgr):
            if self.model is None:
                return None, 0.0
            img = cv2.resize(frame_bgr, (IMG_W, IMG_H))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            pred = self.model.predict(img)
            class_idx = int(np.argmax(pred, axis=1)[0])
            confidence = float(np.max(pred))
            return class_idx, confidence

        def transform(self, frame):
            """Se aplica a cada frame (VideoTransformerBase)"""
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            display_img = img.copy()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detecciones
            motion = self.detect_motion(gray)
            yellow, mask = self.detect_yellow(img)

            class_idx, confidence = (None, 0.0)
            classified = False

            # Lógica: si hay movimiento o amarillo, intentar clasificar (si hay modelo)
            if (motion or yellow) and self.model is not None:
                class_idx, confidence = self.classify_frame(img)
                classified = True

            # Elegir estado prioritario: persona cercana (no hacemos reconocimiento aquí),
            # luego amarillo, luego movimiento
            # Nota: Para entrada/salida usamos el usuario autenticado que vino por login
            if classified and class_idx is not None:
                status = f"Clasificado: Clase {class_idx} ({confidence*100:.1f}%)"
                color = (0, 255, 0) if (confidence >= 0.5) else (0, 165, 255)
            elif yellow:
                status = "Banano (color) detectado"
                color = (0, 255, 0)
            elif motion:
                status = "Movimiento detectado"
                color = (0, 0, 255)
            else:
                status = "Sin actividad"
                color = (255, 255, 255)

            # Dibujar border y texto
            h, w = display_img.shape[:2]
            cv2.rectangle(display_img, (5,5), (w-5, h-5), color, 3)
            cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Si se clasificó con confianza alta, guardar evento
            if classified and class_idx is not None and confidence >= 0.35:
                # Guardar snapshot si pedido
                ts = datetime.utcnow().isoformat().replace(":", "-")
                if save_snapshots:
                    snap_path = SNAP_DIR / f"{self.user}_{ts}.jpg"
                    cv2.imwrite(str(snap_path), display_img)

                # Registrar en CSV
                with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.utcnow().isoformat(), self.user, "clasificacion", class_idx, f"{confidence:.4f}"])

                # Actualizar estado compartido (visible en UI)
                shared["last_status"] = status
                shared["last_class"] = str(class_idx)
                shared["last_conf"] = confidence
                shared["last_time"] = datetime.utcnow().isoformat()

            # Si detecta amarillo sin modelo, registrar evento simple
            if yellow and self.model is None:
                with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.utcnow().isoformat(), self.user, "color_banano", "", ""])

                shared["last_status"] = "Banano (color) detectado"
                shared["last_class"] = ""
                shared["last_conf"] = 0.0
                shared["last_time"] = datetime.utcnow().isoformat()

            return cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    # Ejecutar webrtc streamer si el usuario activó la cámara
    if run_stream:
        ctx = webrtc_streamer(
            key="banana-detector",
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=DetectorTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )
    else:
        st.info("Activa la cámara para comenzar la detección en vivo.")

    # Mostrar información en la columna derecha
    with col_info:
        st.markdown("### Estado actual")
        st.write(f"Usuario: **{st.session_state['user']}**")
        st.write("Último evento:")
        st.write(f"- Estado: {shared['last_status']}")
        st.write(f"- Clase: {shared['last_class']}")
        st.write(f"- Confianza: {shared['last_conf']:.3f}")
        st.write(f"- Hora (UTC): {shared['last_time']}")

        if st.button("Ver registro (últimas 50 entradas)"):
            rows = []
            with open(LOG_CSV, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                rows = reader[-50:]
            st.write(rows)

    # Footer: botón para descargar registro
    st.markdown("---")
    with open(LOG_CSV, "rb") as f:
        csv_bytes = f.read()
    st.download_button("📥 Descargar registro CSV", data=csv_bytes, file_name="registro.csv", mime="text/csv")

# ---------------------------
# Si no autenticado: no mostrar panel principal
# ---------------------------
else:
    st.warning("Accede con un rostro registrado para entrar al panel principal.")
    st.info("Si no tienes una cuenta registrada, añade una imagen a la carpeta 'rostros_registrados/' en el repo y recarga la app.")

