import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import tensorflow as tf

# Configuración inicial
st.set_page_config(
    page_title="🍌 Banana Security System",
    page_icon="🍌",
    layout="wide"
)

# Constantes
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
REGISTERED_FACES_DIR = "registered_faces"
MODEL_PATH = "banana_classifier.h5"
LOG_FILE = "events_log.csv"
SNAPSHOTS_DIR = "snapshots"

# Configuración de detección
MOTION_THRESHOLD = 5000  # Umbral de píxeles cambiados para detección de movimiento
YELLOW_THRESHOLD = 3000  # Umbral de píxeles amarillos para detección de banano
FACE_MATCH_THRESHOLD = 2000  # Umbral MSE para reconocimiento facial

# Configuración de WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Crear directorios necesarios
Path(REGISTERED_FACES_DIR).mkdir(exist_ok=True)
Path(SNAPSHOTS_DIR).mkdir(exist_ok=True)

# Cargar modelo de clasificación (si existe)
@st.cache_resource
def load_model():
    if Path(MODEL_PATH).exists():
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
    return None

model = load_model()

# Funciones de utilidad
def log_event(user, event_type, class_idx="", confidence=""):
    """Registra un evento en el archivo CSV"""
    timestamp = datetime.now().isoformat()
    new_entry = {
        "timestamp": timestamp,
        "user": user,
        "event": event_type,
        "class": class_idx,
        "confidence": confidence
    }
    
    df = pd.DataFrame([new_entry])
    if not Path(LOG_FILE).exists():
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

def detect_yellow(frame):
    """Detecta píxeles amarillos usando HSV"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return np.count_nonzero(mask), mask

def preprocess_for_model(frame, size=(224, 224)):
    """Prepara el frame para el modelo de clasificación"""
    img = cv2.resize(frame, size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# Sistema de reconocimiento facial
class FaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.known_faces = self.load_known_faces()
    
    def load_known_faces(self):
        """Carga rostros conocidos desde el directorio"""
        faces = {}
        for face_file in Path(REGISTERED_FACES_DIR).glob("*.*"):
            if face_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img = cv2.imread(str(face_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces[face_file.stem] = self.extract_face_features(img)
        return faces
    
    def extract_face_features(self, gray_img):
        """Extrae características faciales simples"""
        faces = self.face_cascade.detectMultiScale(gray_img, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_region = gray_img[y:y+h, x:x+w]
        else:
            face_region = gray_img  # Usar toda la imagen si no se detecta cara
        
        face_region = cv2.resize(face_region, (100, 100))
        return face_region.flatten().astype(np.float32)
    
    def recognize_face(self, frame):
        """Reconoce un rostro en el frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_face = self.extract_face_features(gray)
        
        best_match = None
        min_distance = float('inf')
        
        for name, known_face in self.known_faces.items():
            distance = np.mean((known_face - current_face) ** 2)  # MSE
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        if best_match and min_distance < FACE_MATCH_THRESHOLD:
            return best_match, min_distance
        return None, None

# Transformador de video para WebRTC
class BananaDetector(VideoTransformerBase):
    def __init__(self):
        self.prev_gray = None
        self.face_recognizer = FaceRecognizer()
        self.user = None
        self.last_event = "Sin actividad"
    
    def detect_motion(self, gray):
        """Detecta movimiento comparando con el frame anterior"""
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        
        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_pixels = np.count_nonzero(thresh)
        self.prev_gray = gray
        
        return motion_pixels > MOTION_THRESHOLD
    
    def transform(self, frame):
        """Procesa cada frame del video"""
        frame = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Estado inicial
        status = "Sin actividad"
        status_color = (255, 255, 255)  # Blanco
        classified = False
        class_idx = ""
        confidence = 0.0
        
        # Detecciones
        motion = self.detect_motion(gray)
        yellow_pixels, yellow_mask = detect_yellow(frame)
        yellow = yellow_pixels > YELLOW_THRESHOLD
        
        # Reconocimiento facial (solo si no hay usuario)
        if self.user is None:
            user, distance = self.face_recognizer.recognize_face(frame)
            if user:
                self.user = user
                log_event(user, "login")
                status = f"Usuario reconocido: {user}"
                status_color = (0, 255, 0)  # Verde
        
        # Clasificación si hay modelo y actividad
        if model and (motion or yellow) and self.user:
            processed_img = preprocess_for_model(frame)
            preds = model.predict(processed_img)
            class_idx = np.argmax(preds)
            confidence = float(np.max(preds))
            classified = True
            
            if confidence > 0.5:  # Solo considerar predicciones confiables
                status = f"Clase {class_idx} ({confidence:.1%})"
                status_color = (0, 255, 0)  # Verde
                log_event(self.user, "classification", class_idx, confidence)
                
                # Guardar snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = f"{SNAPSHOTS_DIR}/{self.user}_{timestamp}.jpg"
                cv2.imwrite(snapshot_path, frame)
        
        elif yellow:
            status = "Banano detectado (color)"
            status_color = (0, 255, 255)  # Amarillo
            if self.user:
                log_event(self.user, "yellow_detected")
        
        elif motion:
            status = "Movimiento detectado"
            status_color = (0, 0, 255)  # Rojo
            if self.user:
                log_event(self.user, "motion_detected")
        
        # Dibujar información en el frame
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), 
                     status_color, 2)
        
        # Actualizar último evento para la UI
        self.last_event = status
        
        return frame

# Interfaz de usuario
def main():
    st.title("🍌 Banana Security System")
    st.markdown("Sistema de reconocimiento facial y detección de bananos")
    
    # Barra lateral
    st.sidebar.title("Configuración")
    
    # Subir modelo
    uploaded_model = st.sidebar.file_uploader("Subir modelo (.h5)", type=["h5"])
    if uploaded_model is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.sidebar.success("Modelo subido correctamente")
        st.experimental_rerun()
    
    # Subir rostros autorizados
    st.sidebar.subheader("Registrar nuevos usuarios")
    new_face = st.sidebar.camera_input("Tomar foto para registro")
    if new_face is not None:
        user_name = st.sidebar.text_input("Nombre del usuario")
        if user_name:
            img = Image.open(new_face)
            img.save(f"{REGISTERED_FACES_DIR}/{user_name}.jpg")
            st.sidebar.success(f"Usuario {user_name} registrado")
    
    # Mostrar usuarios registrados
    st.sidebar.subheader("Usuarios registrados")
    registered_users = [f.stem for f in Path(REGISTERED_FACES_DIR).glob("*.*")]
    if registered_users:
        for user in registered_users:
            st.sidebar.write(f"- {user}")
    else:
        st.sidebar.write("No hay usuarios registrados")
    
    # Área principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Cámara en vivo")
        ctx = webrtc_streamer(
            key="banana-detector",
            video_transformer_factory=BananaDetector,
            rtc_configuration=RTC_CONFIGURATION,
            async_transform=True
        )
        
        if ctx.video_transformer:
            detector = ctx.video_transformer
            st.subheader("Estado actual")
            st.write(f"**Usuario:** {detector.user or 'No identificado'}")
            st.write(f"**Estado:** {detector.last_event}")
            
            if st.button("Forzar reconocimiento facial"):
                detector.user = None
    
    with col2:
        st.subheader("Registro de eventos")
        
        if Path(LOG_FILE).exists():
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df.tail(10))
            
            with open(LOG_FILE, "rb") as f:
                st.download_button(
                    "Descargar registro completo",
                    f.read(),
                    file_name="events_log.csv",
                    mime="text/csv"
                )
        else:
            st.write("No hay eventos registrados aún")
        
        st.subheader("Snapshots guardados")
        snapshots = list(Path(SNAPSHOTS_DIR).glob("*.jpg"))
        if snapshots:
            latest_snapshot = max(snapshots, key=os.path.getmtime)
            st.image(str(latest_snapshot), caption="Último snapshot")
        else:
            st.write("No hay snapshots guardados")

if __name__ == "__main__":
    main()
