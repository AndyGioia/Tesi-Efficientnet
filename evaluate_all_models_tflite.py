"""
Valuta i modelli TFLite sul test set del FER e confronta l'accuracy con i modelli .keras
per ogni modello TFLite viene:
- caricato l'interpreter TFLite
- scopre l'input shape atteso
- preparato il test set con relativo preprocessing
- eseguita l'inferenza immagine per immagine (batch = 1)
- calcolate accuracy, F1 e latenza CPU
- confrontati i risultati con il keras
"""
import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Configurazione
BASE_DIR = "/mnt/i/TESI/Tesi-Efficientnet"
TFLITE_DIR = os.path.join(BASE_DIR, "tflite_models")
DATA_DIR = os.path.join(BASE_DIR, "FER", "images")
RESULTS_DIR = os.path.join(BASE_DIR, "tflite_evaluation")
os.makedirs(RESULTS_DIR, exist_ok = True)

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

#accuracy originale del keras per riferimento
KERAS_ACCURACY = {
    "EfficientNetV2B2": 70.02,
    "EfficientNetB0": 69.88,
    "MobileNetV3Small": 67.61
}

MODELS = [
    {
        "name": "EfficientNetV2B2",
        "tflite_path": os.path.join(TFLITE_DIR, "EfficientNetV2B2_dynamic_range.tflite"),
        "img_size": 260,
    },
    {
        "name": "EfficientNetB0",
        "tflite_path": os.path.join(TFLITE_DIR, "EfficientNetB0_dynamic_range.tflite"),
        "img_size": 224,
    },
    {
        "name": "MobileNetV3Small",
        "tflite_path": os.path.join(TFLITE_DIR, "MobileNetV3Small_dynamic_range.tflite"),
        "img_size": 224,
    },

]

#Carica e ispeziona l'interpreter TFLite
def load_interpreter(tflite_path, name):
    #step diagnostico, carica il modello tflite e stampa i dettagli di input/output
    print(f"[1/5] Caricamento interpreter: {name}")
    interpreter = tf.lite.Interpreter(model_path = tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f" Input shape: {input_details[0]['shape']}")
    print(f" Input dtype: {input_details[0]['dtype']}")
    print(f" Output shape: {output_details[0]['shape']}")
    print(f" Output dtype: {output_details[0]['dtype']}")

    expected_input_channels = 1
    actual_channels = input_details[0]['shape'][-1]
    if actual_channels != expected_input_channels:
        print(f" ATTENZIONE - Input ha {actual_channels} canali invece di {expected_input_channels}!")
        print(f" Impostare USE_RGB_INPUT = True")
    else:
        print(f" Input shape corretto: {actual_channels} canale grayscale")
    
    return interpreter, input_details, output_details

#Preparazione del test set
def prepare_test_set(img_size, batch_size = 32):
    #creazione del generatore per il test set
    datagen = ImageDataGenerator() #nessun rescale
    test_set = datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size = (img_size, img_size),
        color_mode = "grayscale", #1 canale, come nel training
        batch_size = batch_size,
        class_mode = "categorical",
        classes = CLASSES,
        shuffle = False #per metriche corrette
    )
    print(f" Test set: {test_set.n} immagini, {test_set.num_classes} classi")
    return test_set

#Inferenza immagine per immagine (Batch = 1)
def run_inference(interpreter, input_details, output_details, test_set):
    #esegue l'inferenza sul test set, una immagine alla volta
    #avendo input shape fisso, TFLite non supporta batch dinamici, ogni call processa 1 immagine
    print(f"\n[3/5] Inferenza sul test set ({test_set.n} immagini)...")
    #verifica che il grafo abbia inglobato il Concatenate
    use_rgb = (input_details[0]['shape'][-1] == 3)
    in_dtype = input_details[0]['dtype']

    y_pred_all = []
    y_true_all = list(test_set.classes)
    latencies = []
    test_set.reset() #reset del generatore
    processed = 0

    while processed < test_set.n:
        batch_images, _ = next(test_set)
        for img in batch_images:
            if processed >= test_set.n:
                break

            img_input = np.expand_dims(img, axis = 0) #img ha shape (H,W,1) viene aggiunta la batch per avere (1,H,W,1)

            if use_rgb: #nel caso il TFLite si aspettasse 3 canali
                img_input = np.repeat(img_input, 3, axis = -1) #(1,H,W,3)
            img_input = img_input.astype(in_dtype) #converte al dtype atteso dall'interpreter
            #esecuzione dell'inferenza e misurazione della latenza
            t0 = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            latencies.append((time.perf_counter() - t0) * 1000)

            y_pred_all.append(np.argmax(output[0]))
            processed += 1

            if processed % 200 == 0:
                print(f" ... {processed}/{test_set.n}")
    
    lat_mean = np.mean(latencies)
    lat_p95 = np.percentile(latencies, 95)
    print(f"  Inferenza completata")
    print(f"  Latenza media: {lat_mean:.1f} ms/img")
    print(f"  Latenza p95: {lat_p95:.1f} ms/img")
    return np.array(y_pred_all), np.array(y_true_all), lat_mean, lat_p95

#Calcolo delle metriche e confronto col .keras
def compute_metrics(y_pred, y_true, name, lat_mean, lat_p95):
    print(f"[4/5] Metriche - {name}")
    accuracy = np.mean(y_pred == y_true) * 100
    keras_acc = KERAS_ACCURACY.get(name, None)
    delta = accuracy - keras_acc if keras_acc else None
    print(f"  TFLite accuracy: {accuracy:.2f}%")
    if keras_acc:
        print(f"  Keras accuracy: {keras_acc:.2f}%")
        print(f"  Delta: {delta:+.2f} pp")
    
    if delta is not None and delta < -5:
        print(f"  DELTA TROPPO NEGATIVO ({delta:.2f} pp)! POSSIBILI CAUSE:")
        print(f" 1. Preprocessing errato (rescale o dtype errato)")
        print(f" 2 Canali input sbagliati (RGB invece di Grayscale)")
        print(f" 3 Errore nel caricamento del file .tflite")
    report = classification_report(y_true, y_pred, target_names = CLASSES, digits = 4)
    print(f"\n{report}")

    return accuracy, delta, report

#Salvataggio di Confusion Matrix e del Report
def save_outputs(y_pred, y_true, name, accuracy, delta, report, lat_mean, lat_p95):
    print(f"\n[5/5] Salvataggio output - {name}")

    #Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize = (9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = CLASSES)
    disp.plot(ax = ax, cmap = "Blues", colorbar = False, xticks_rotation = 45)
    delta_str = f"{delta:+.2f} pp vs Keras" if delta else ""
    ax.set_title(f"Confusion Matrix TFLite - {name}\n Accuracy: {accuracy:.2f}% ({delta_str})", fontsize = 12)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"cm_tflite_{name}_{TIMESTAMP}.png")
    plt.savefig(cm_path, dpi = 150)
    plt.close()
    print(f" Confusion Matrix: {cm_path}")

    #Report Testuale
    report_path = os.path.join(RESULTS_DIR, f"report_tflite_{name}_{TIMESTAMP}.txt")
    with open(report_path, "w") as f:
        f.write(f"Modello TFLite: {name}\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"TFLite Accuracy: {accuracy:.2f}%\n")
        keras_acc = KERAS_ACCURACY.get(name)
        if keras_acc:
            f.write(f"Keras Accuracy: {keras_acc:.2f}%\n")
            f.write(f"Delta: {delta:.2f} pp\n")
        f.write(f"Latenza media CPU: {lat_mean:.1f} ms\n")
        f.write(f"Latenza p95 CPU: {lat_p95:.1f} ms\n\n")
        f.write(report)
    print(f"  Report: {report_path}")

    #JSON di riepilogo
    summary = {
        "model_name": name,
        "tflite_accuracy": round(accuracy, 2),
        "keras_accuracy": KERAS_ACCURACY.get(name),
        "delta_pp": round(delta, 2) if delta else None,
        "latency_mean_ms": round(lat_mean, 1),
        "latency_p95_ms": round(lat_p95, 1)
    }
    json_path = os.path.join(RESULTS_DIR, f"summary_tflite_{name}_{TIMESTAMP}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent = 2)
    print(f" File JSON: {json_path}")

    return summary

#main della funzione
if __name__ == "__main__":
    all_summaries = []
    for cfg in MODELS:
        if not os.path.exists(cfg["tflite_path"]):
            print(f"\n File TFLite non presente: {cfg['tflite_path']} - salto")
            continue
        name = cfg["name"]
        print(f"\n\n{'#'*60}")
        print(f" VALUTAZIONE TFLITE: [name]")
        print(f"{'#'*60}")
        
        interpreter, input_details, output_details = load_interpreter(cfg["tflite_path"], name)
        print(f"[2/5] Preparazione test set (img_size = {cfg['img_size']})...")
        test_set = prepare_test_set(cfg["img_size"])

        y_pred, y_true, lat_mean, lat_p95 = run_inference(interpreter, input_details, output_details, test_set)
        accuracy, delta, report = compute_metrics(y_pred, y_true, name, lat_mean, lat_p95)
        
        summary = save_outputs(y_pred, y_true, name, accuracy, delta, report, lat_mean, lat_p95)
        all_summaries.append(summary)
    
    #riepilogo finale comparativo
    print(f"\n\n{'='*65}")
    print(f"CONFRONTO FINALE TRA KERAS E TFLITE DYNAMIC RANGE")
    print(f"{'='*65}")
    print(f" {'Modello':<22} {'Keras':>8} {'Delta':>8} {'Lat':>8}")
    print(f"{'-'*65}")
    for s in all_summaries:
        print(f"  {s['model_name']:<22} "
              f"{str(s['keras_accuracy'])+'%':>8} "
              f"{str(s['tflite_accuracy'])+'%':>8} "
              f"{(str(s['delta_pp'])+'pp') if s['delta_pp'] else 'N/A':>8} "
              f"{str(s['latency_mean_ms'])+'ms':>8}")
    print(f"\n Output salvato in {RESULTS_DIR}")
