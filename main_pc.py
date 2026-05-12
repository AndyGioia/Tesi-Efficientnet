import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import os


# ── Configurazione modelli FERPlus ──────────────────────────────────────────
MODELS_CONFIG = {
    "1": {
        "name": "EfficientNetB0",
        "path": "I:/TESI/Tesi-Efficientnet/models/efficientnetb0_ferplus/best_model_b0_ferplus_ft.keras",
        "input_size": (224, 224),
        "channels": 1,
    },
    "2": {
        "name": "EfficientNetV2B2",
        "path": "I:/TESI/Tesi-Efficientnet/models/efficientnetb2v2_ferplus/best_model_v2b2_ferplus_ft.keras",
        "input_size": (260, 260),
        "channels": 1,
    },
    "3": {
        "name": "MobileNetV3Small",
        "path": "I:/TESI/Tesi-Efficientnet/models/mobilenetv3small_ferplus/best_model_mobilenetv3small_ferplus_ft.keras",
        "input_size": (224, 224),
        "channels": 1,
    },
}

EMOTION_LABELS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_COLORS = {
    'angry':    (0,   0,   220),
    'contempt': (0,   140, 255),
    'disgust':  (0,   180, 0),
    'fear':     (180, 0,   180),
    'happy':    (0,   220, 220),
    'neutral':  (200, 200, 200),
    'sad':      (220, 100, 0),
    'surprise': (0,   200, 255),
}

HAAR_PATH     = "haarcascade_frontalface_default.xml"
DNN_PROTOTXT  = "I:/TESI/Tesi-Efficientnet/dnn_face/deploy.prototxt"
DNN_MODEL     = "I:/TESI/Tesi-Efficientnet/dnn_face/res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONF_THRESH = 0.5

FONT = cv2.FONT_HERSHEY_TRIPLEX


# ── Helper: testo con sfondo scuro semitrasparente ───────────────────────────
def put_text_bg(frame, text, pos, font, scale, color, thickness=1, bg_alpha=0.55, pad=4):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, pos, font, scale, color, thickness)


# ── DNN detector ─────────────────────────────────────────────────────────────
def load_dnn_detector():
    if not os.path.exists(DNN_PROTOTXT) or not os.path.exists(DNN_MODEL):
        print("[WARN] File DNN non trovati, passaggio a Haar.")
        return None
    net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_MODEL)
    print("[OK] DNN face detector caricato.")
    return net


def detect_faces_dnn(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                  (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < DNN_CONF_THRESH:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces


def detect_faces_haar(classifier, frame, is_photo=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    neighbors = 6 if is_photo else 5
    min_size  = (60, 60) if is_photo else (48, 48)
    return list(classifier.detectMultiScale(
        gray,
        scaleFactor=1.1 if is_photo else 1.3,
        minNeighbors=neighbors,
        minSize=min_size
    ))


# ── Menu principale modelli ──────────────────────────────────────────────────
def draw_menu(selected=None):
    img = np.zeros((420, 560, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    cv2.putText(img, "FERPlus Emotion Detector", (60, 50),
                FONT, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "Seleziona un modello:", (80, 95),
                FONT, 0.65, (180, 180, 180), 1)

    options = [
        ("1", "EfficientNetB0"),
        ("2", "EfficientNetV2B2"),
        ("3", "MobileNetV3Small"),
    ]
    for i, (key, label) in enumerate(options):
        y = 155 + i * 70
        color = (0, 200, 100) if selected == key else (60, 60, 60)
        cv2.rectangle(img, (60, y - 30), (500, y + 20), color, -1)
        cv2.rectangle(img, (60, y - 30), (500, y + 20), (100, 100, 100), 1)
        cv2.putText(img, f"[{key}]  {label}", (75, y + 2),
                    FONT, 0.55, (255, 255, 255), 1)

    cv2.putText(img, "Premi Q per uscire", (170, 390),
                FONT, 0.55, (100, 100, 100), 1)
    return img


# ── Menu modalita ────────────────────────────────────────────────────────────
def draw_mode_menu(model_name, use_dnn, selected=None):
    img = np.zeros((380, 600, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    cv2.putText(img, f"Modello: {model_name}", (50, 42),
                FONT, 0.75, (200, 200, 200), 1)

    detector_label = "DNN (ResNet SSD)" if use_dnn else "Haar Cascade"
    detector_color = (0, 220, 150) if use_dnn else (180, 180, 0)
    cv2.putText(img, f"Detector: {detector_label}  [D=cambia]", (50, 78),
                FONT, 0.55, detector_color, 1)

    cv2.line(img, (40, 95), (560, 95), (70, 70, 70), 1)
    cv2.putText(img, "Seleziona modalita:", (50, 125),
                FONT, 0.65, (180, 180, 180), 1)

    options = [
        ("w", "W  -  Webcam live"),
        ("f", "F  -  Carica foto"),
    ]
    for i, (key, label) in enumerate(options):
        y = 190 + i * 75
        color = (0, 180, 220) if selected == key else (60, 60, 60)
        cv2.rectangle(img, (50, y - 30), (550, y + 25), color, -1)
        cv2.rectangle(img, (50, y - 30), (550, y + 25), (100, 100, 100), 1)
        cv2.putText(img, label, (80, y + 5),
                    FONT, 0.65, (255, 255, 255), 1)

    cv2.putText(img, "M=menu modelli   Q=esci   D=cambia detector", (70, 355),
                FONT, 0.42, (100, 100, 100), 1)
    return img


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_face(face_roi, input_size, channels):
    face_resized = cv2.resize(face_roi, input_size, interpolation=cv2.INTER_AREA)
    if channels == 1:
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_array = face_gray.astype(np.float32)
        face_array = np.expand_dims(face_array, axis=-1)
    else:
        face_array = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(face_array, axis=0)


# ── Disegna predizione con sfondo ────────────────────────────────────────────
def draw_prediction(frame, x, y, w, h, label, confidence):
    color = EMOTION_COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} ({confidence*100:.1f}%)"
    text_y = y - 10 if y - 10 > 10 else y + h + 22
    put_text_bg(frame, text, (x, text_y), FONT, 0.75, color, thickness=1, bg_alpha=0.6)


# ── Processa volti trovati ───────────────────────────────────────────────────
def process_faces(frame, faces, model, input_size, channels):
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue
        tensor = preprocess_face(face_roi, input_size, channels)
        preds = model.predict(tensor, verbose=0)[0]
        idx = np.argmax(preds)
        draw_prediction(frame, x, y, w, h, EMOTION_LABELS[idx], preds[idx])


# ── Modalita WEBCAM ──────────────────────────────────────────────────────────
def run_webcam(model, model_cfg, haar_classifier, dnn_net, use_dnn):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRORE] Webcam non trovata.")
        return "mode_menu"

    input_size = model_cfg["input_size"]
    channels   = model_cfg["channels"]
    model_name = model_cfg["name"]
    det_label  = "DNN" if use_dnn else "Haar"
    window_name = f"FERPlus - {model_name} [{det_label}] | M=menu Q=esci"

    print(f"[INFO] Webcam attiva - {model_name} con {det_label}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_dnn and dnn_net is not None:
            faces = detect_faces_dnn(dnn_net, frame)
        else:
            faces = detect_faces_haar(haar_classifier, frame, is_photo=False)

        process_faces(frame, faces, model, input_size, channels)

        put_text_bg(frame, f"{model_name} [{det_label}]",
                    (10, 25), FONT, 0.6, (255, 255, 255), thickness=1, bg_alpha=0.6)
        put_text_bg(frame, "M=menu  Q=esci  D=detector",
                    (10, 52), FONT, 0.5, (200, 200, 200), thickness=1, bg_alpha=0.6)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            cap.release(); cv2.destroyAllWindows(); return "quit"
        elif key == ord('m') or key == ord('M'):
            cap.release(); cv2.destroyAllWindows(); return "mode_menu"

    cap.release(); cv2.destroyAllWindows(); return "quit"


# ── Modalita FOTO ────────────────────────────────────────────────────────────
def run_photo(model, model_cfg, haar_classifier, dnn_net, use_dnn):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    img_path = filedialog.askopenfilename(
        title="Seleziona un'immagine",
        filetypes=[("Immagini", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tutti i file", "*.*")]
    )
    root.destroy()

    if not img_path:
        print("[INFO] Nessuna immagine selezionata.")
        return "mode_menu"

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ERRORE] Impossibile leggere: {img_path}")
        return "mode_menu"

    input_size = model_cfg["input_size"]
    channels   = model_cfg["channels"]
    model_name = model_cfg["name"]
    det_label  = "DNN" if use_dnn else "Haar"

    if use_dnn and dnn_net is not None:
        faces = detect_faces_dnn(dnn_net, frame)
    else:
        faces = detect_faces_haar(haar_classifier, frame, is_photo=True)

    if len(faces) == 0:
        print("[INFO] Nessun volto rilevato.")
        put_text_bg(frame, "Nessun volto rilevato", (20, 50),
                    FONT, 1.0, (0, 80, 255), thickness=2, bg_alpha=0.65)
    else:
        process_faces(frame, faces, model, input_size, channels)
        print(f"[INFO] Rilevati {len(faces)} volto/i con {det_label}.")

    max_dim = 900
    h_img, w_img = frame.shape[:2]
    if max(h_img, w_img) > max_dim:
        scale = max_dim / max(h_img, w_img)
        frame = cv2.resize(frame, (int(w_img * scale), int(h_img * scale)))

    put_text_bg(frame, f"{model_name} [{det_label}] | Premi un tasto per continuare",
                (10, 25), FONT, 0.55, (255, 255, 255), thickness=1, bg_alpha=0.6)

    cv2.imshow(f"FERPlus - {model_name} - Foto", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return "mode_menu"


# ── Show mode menu ───────────────────────────────────────────────────────────
def show_mode_menu(model, model_cfg, haar_classifier, dnn_net):
    model_name = model_cfg["name"]
    use_dnn = (dnn_net is not None)
    selected = None

    cv2.namedWindow("FERPlus - Modalita")

    while True:
        cv2.imshow("FERPlus - Modalita", draw_mode_menu(model_name, use_dnn, selected))
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows(); return "quit"
        elif key == ord('m') or key == ord('M'):
            cv2.destroyAllWindows(); return "menu"
        elif key == ord('d') or key == ord('D'):
            if dnn_net is None:
                print("[WARN] DNN non disponibile.")
            else:
                use_dnn = not use_dnn
                print(f"[INFO] Detector: {'DNN' if use_dnn else 'Haar'}")
        elif key == ord('w') or key == ord('W'):
            selected = "w"
            cv2.imshow("FERPlus - Modalita", draw_mode_menu(model_name, use_dnn, selected))
            cv2.waitKey(300); cv2.destroyAllWindows()
            result = run_webcam(model, model_cfg, haar_classifier, dnn_net, use_dnn)
            if result == "quit": return "quit"
            cv2.namedWindow("FERPlus - Modalita"); selected = None
        elif key == ord('f') or key == ord('F'):
            selected = "f"
            cv2.imshow("FERPlus - Modalita", draw_mode_menu(model_name, use_dnn, selected))
            cv2.waitKey(300); cv2.destroyAllWindows()
            result = run_photo(model, model_cfg, haar_classifier, dnn_net, use_dnn)
            if result == "quit": return "quit"
            cv2.namedWindow("FERPlus - Modalita"); selected = None


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    tf.get_logger().setLevel('ERROR')

    haar_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_PATH)
    if haar_classifier.empty():
        print("[ERRORE] Haar cascade non trovato.")
        return

    dnn_net = load_dnn_detector()
    loaded_models = {}
    action = "menu"

    while action == "menu":
        cv2.namedWindow("FERPlus - Menu")
        selected_key = None

        while True:
            cv2.imshow("FERPlus - Menu", draw_menu(selected_key))
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows(); print("[INFO] Uscita."); return

            if key != 255 and chr(key) in MODELS_CONFIG:
                selected_key = chr(key)
                for _ in range(10):
                    cv2.imshow("FERPlus - Menu", draw_menu(selected_key))
                    cv2.waitKey(30)
                cv2.destroyAllWindows()
                break

        if selected_key is None:
            continue

        cfg = MODELS_CONFIG[selected_key]

        if selected_key not in loaded_models:
            print(f"\n[INFO] Caricamento {cfg['name']}... (prima volta, attendi)")
            try:
                loaded_models[selected_key] = load_model(cfg["path"])
                print("[OK] Modello caricato.")
            except Exception as e:
                print(f"[ERRORE] {e}")
                action = "menu"; continue
        else:
            print(f"\n[INFO] {cfg['name']} gia in cache.")

        model = loaded_models[selected_key]
        action = show_mode_menu(model, cfg, haar_classifier, dnn_net)

    print("[INFO] Programma terminato.")


if __name__ == "__main__":
    main()