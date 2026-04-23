"""
Converte i modelli in formato TFLite con quantizzazione Dynamic Range
Al fine di minimizzare la perdita di accuratezza dei modelli si è optato per la quantizzazione Dynamic Range al posto dell'INT8 statico:
il Dynamic Range non richiede un representative dataset per la conversione ma quantizza solo i pesi in INT8, mantenendo le attivazioni in Float32
perdita di accuratezza di circa 1-2 punti con una riduzione delle dimensioni dei modelli del 75% circa
al fine di evitare problemi nella conversione dovuti al concatenate e alla natura grayscale del FER Dataset col quale sono stati addestrati i modelli
la conversione partirà dal SavedModel al posto del .keras originale, permettendo al converter di tracciare il layer concatenate senza problemi
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Configurazione
BASE_DIR = "/mnt/i/TESI/Tesi-Efficientnet"
OUTPUT_DIR = os.path.join(BASE_DIR, "tflite_models")
os.makedirs(OUTPUT_DIR, exist_ok = True)

#Definizione dei modelli e le loro rispettive configurazioni
MODELS = [
    {
        "name": "EfficientNetV2B2",
        "keras_path": os.path.join(BASE_DIR, "models", "efficientnetb2v2", "best_model_v2b2_ft.keras"),
        "img_size": 260, #dimensione input nativa del backbone
        "expected_keras_acc": 70.02, #valore di riferimento della test accuracy del modello
    },

    {
        "name": "EfficientNetB0",
        "keras_path": os.path.join(BASE_DIR, "models", "efficientnetb0", "best_model_b0_ft.keras"),
        "img_size": 224,
        "expected_keras_acc": 68.46, 

    },
    {
        "name": "MobileNetV3Small",
        "keras_path": os.path.join(BASE_DIR, "models", "mobilenetv3small", "best_model_mobilenetv3small_ft.keras"),
        "img_size": 224,
        "expected_keras_acc": 64.99,
    },

]

#Funzione principale di conversione
def convert_model(cfg):
    name = cfg["name"]
    keras_path = cfg["keras_path"]
    img_size = cfg["img_size"]

    print(f"\n{'='*60}")
    print(f"  CONVERSIONE: {name}")
    print(f"\n{'='*60}")

    #Caricamento del modello .keras
    print(f"\n[1/4] Caricamento modello da: {keras_path}")
    model = load_model(keras_path)
    print(f"  Parametri totali: {model.count_params():,}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    keras_size_mb = os.path.getsize(keras_path) / (1024 * 1024)
    print(f"  Dimensione .keras su disco: {keras_size_mb:.2f} MB")

    #Esportazione del modello come SaveModel
    #permette al converter di non avere problemi con il concatenate layer che, nel savemodel, viene già compilato
    saved_model_path = os.path.join(OUTPUT_DIR, f"saved_model_{name}")
    print(f"\n[2/4] Esportazione SavedModel in: {saved_model_path}")
    t0 = time.time()
    model.export(saved_model_path)
    print(f" SavedModel esportato in {time.time() - t0:.1f} secondi")

    #Conversione del modello in TFLite con Dynamic Range Quantization
    #quantizza solo i pesi in INT8, le attivazioni restano in Float32
    #maggiore compatibilità con CPU ARM
    print(f"\n[3/4] Conversione TFLite (Dynamic Range)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] #quantizzazione in Dynamic Range
    t0 = time.time()
    tflite_model = converter.convert()
    print(f"  Conversione TFLite completata in {time.time() - t0:.1f} secondi")

    #Salvataggio del modello .tflite
    tflite_path = os.path.join(OUTPUT_DIR, f"{name}_dynamic_range.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    reduction = (1 - tflite_size_mb / keras_size_mb) * 100

    print(f"\n[4/4] File salvato: {tflite_path}")
    print(f"  Dimensione .keras: {keras_size_mb:.2f} MB")
    print(f"  Dimensione .tflite: {tflite_size_mb:.2f} MB")
    print(f"  Riduzione: {reduction:.1f}%")

    expected_tflite_mb = keras_size_mb * 0.26 #stima: 74% di riduzione
    if abs(tflite_size_mb - expected_tflite_mb) > expected_tflite_mb * 0.5:
        print(f"  Dimensione inattesa! Atteso ~{expected_tflite_mb:.2f} MB")
    else:
        print("  Dimensione nella norma")
    
    return tflite_path, keras_size_mb, tflite_size_mb

#Esecuzione
if __name__ == "__main__":
    results = []
    for cfg in MODELS:
        if not os.path.exists(cfg["keras_path"]):
            print(f"  File non trovato: {cfg['keras_path']} - saltato")
            continue
        tflite_path, keras_mb, tflite_mb = convert_model(cfg)
        results.append({
            "name": cfg["name"],
            "keras_mb": keras_mb,
            "tflite_mb": tflite_mb,
            "path": tflite_path
        })

    print(f"\n\n{'='*60}")
    print(f"  RIEPILOGO CONVERSIONI")
    print(f"\n\n{'='*60}")
    print(f"  {'Modello':<22} {'Keras':>8} {'TFLite':>8} {'Riduz.':>8}")
    print(f"  {'-'*50}")
    for r in results:
        red = (1 - r['tflite_mb'] / r['keras_mb']) * 100
        print(f" {r['name']:<22} {r['keras_mb']:>7.2f}MB {r['tflite_mb']:>7.2f}MB {red:>7.1f}%")
    print(f"\n File salvati in: {OUTPUT_DIR}")

    