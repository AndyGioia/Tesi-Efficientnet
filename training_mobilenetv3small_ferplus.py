"""
Addestra il modello per il riconoscimento delle emozioni facciali MobileNetV3-Small, pre-addestrato su ImageNet, sul dataset FERPlus
il dataset FERPlus comprende 8 classi: le 7 di Ekman (angry, disgust, fear, happy, neutral, sad, surprise) + contempt
Le immagini sono 112x112 pixel grayscale (vs 48x48 di FER2013)
il training set include data augmentation pre-applicata offline per bilanciare le classi minoritarie:
- train: 66.379 immagini (28.558 originali + 37.821 augmentate offline)
- validation: 8.341 immagini (3.579 originali + 4.762 augmentate offline)
- test: 3.573 immagini (solo originali, nessuna augmentazione)
al fine dell'addestramento il modello è stato adattato per ricevere l'input grayscale tramite la duplicazione dei canali
L'addestramento avviene tramite Transfer Learning a due fasi:
- Fase 1 (Head Training): viene addestrato solo il classificatore finale, lasciando il backbone del modello congelato
- Fase 2 (Fine Tuning): viene scongelato il backbone e addestrato a un learning rate decisamente minore rispetto alla head
architettura Head: GlobalAveragePooling2D → Dense(1024, relu) → BatchNorm → Dropout(0.3) → Dense(8, softmax)
"""
#Configurazione degli Import
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random
import time
import seaborn as sns
import tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import os


#Caricamento del modello e adattamento a input grayscale
input_shape = (224, 224, 3)  #MobileNetV3Small si aspetta RGB input

input_tensor = Input(shape=(224, 224, 1))  #Input per immagini grayscale
x = concatenate([input_tensor, input_tensor, input_tensor], axis=-1)

#Caricamento del modello pre-addestrato privo del top Layer e adattamento per l'input a 3 canali (grayscale duplicato su 3 canali)
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
x = base_model(x, training=False)  #Usa il modello base in modalità inference
print("MobileNetV3Small output shape:", base_model.output_shape)


#Global Average Pooling
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  #Layer Dense aggiuntivo per portare i parametri a ~1.53M
x = BatchNormalization()(x)             #Normalizzazione per stabilizzare il training della Head
x = Dropout(0.3)(x)                     #Dropout per ridurre l'overfitting
output_layer = Dense(8, activation='softmax')(x)  # 8 classi FERPlus (aggiunge contempt)


#Create the model
model = Model(inputs=input_tensor, outputs=output_layer)
model.summary()


#Percorsi dataset e modelli
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "FERPlus")  # Dataset FERPlus con struttura train/validation/test
MODEL_DIR = os.path.join(BASE_DIR, "models", "mobilenetv3small_ferplus")
os.makedirs(MODEL_DIR, exist_ok=True)


#Data Augmentation
#Le immagini di train e validation contengono già augmentation offline applicata dal dataset.
#Si mantiene comunque una leggera augmentation on-the-fly sul train per variabilità aggiuntiva.
datagen_train = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.25,
    brightness_range=(0.8, 1.2),
    shear_range=0.1,
    fill_mode='nearest'
)

datagen_validation = ImageDataGenerator()
datagen_test = ImageDataGenerator()


#Classi FERPlus (8 classi in ordine alfabetico — corrisponde all'ordine delle cartelle)
CLASSES_FERPLUS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


#Train set
#Le immagini FERPlus sono 112x112 — vengono ridimensionate a 224x224 da target_size
train_set = datagen_train.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=CLASSES_FERPLUS,
    shuffle=True
)

#Validation set
validation_set = datagen_validation.flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=CLASSES_FERPLUS,
    shuffle=False
)

#Test set
test_set = datagen_test.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=CLASSES_FERPLUS,
    shuffle=False
)


#Compute class weights dynamically
def get_class_weights(generator):
    class_indices = generator.class_indices
    num_classes = len(class_indices)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=generator.classes
    )
    return dict(enumerate(class_weights))


class_weight_dict = get_class_weights(train_set)
class_weight_dict = {k: min(v, 4.0) for k, v in class_weight_dict.items()}  # Limita il peso massimo a 4.0
print("Class weights (clipped):", class_weight_dict)


#FASE 1: Training della Head


checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model_mobilenetv3small_ferplus.keras"),
    monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)

base_model.trainable = False  # Congela i pesi del modello base
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.10),
    metrics=['accuracy']
)
print("\n === FASE 1: Training HEAD ===")

start_total = time.time()
start_fase1 = time.time()

history_head = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=5,
    callbacks=[checkpoint],
    class_weight=class_weight_dict
)

end_fase1 = time.time()
print(f"Tempo impiegato per FASE 1: {(end_fase1 - start_fase1)/60:.1f} minuti")


#FASE 2: Fine-tuning del modello completo


checkpoint_ft = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model_mobilenetv3small_ferplus_ft.keras"),
    monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
early_stopping_ft = EarlyStopping(
    monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True
)
reduce_lr_ft = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.3,
    patience=4,
    verbose=1,
    min_delta=0.001
)
callbacks_ft = [early_stopping_ft, checkpoint_ft, reduce_lr_ft]

print("\n === FASE 2: Fine-Tuning ===")

start_fase2 = time.time()

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=AdamW(learning_rate=0.00003, weight_decay=0.0001),
    loss=CategoricalCrossentropy(label_smoothing=0.06),
    metrics=['accuracy']
)
history_ft = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=40,
    callbacks=callbacks_ft,
    class_weight=class_weight_dict
)

end_fase2 = time.time()
end_total = time.time()
print(f"Tempo impiegato per FASE 2: {(end_fase2 - start_fase2)/60:.1f} minuti")
print(f"Tempo totale di addestramento: {(end_total - start_total)/60:.1f} minuti")


#Valutazione Finale sul TEST SET
print("\n === VALUTAZIONE FINALE SUL TEST SET ===")
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


#Confusion Matrix e Classification Report
from sklearn.metrics import classification_report, confusion_matrix

class_names = CLASSES_FERPLUS

print("\n === CONFUSION MATRIX E METRICHE PER CLASSE ===")
test_set.reset()
y_pred_probs = model.predict(test_set, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_set.classes

#Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

#Salva report in txt
report_path = os.path.join(MODEL_DIR, f"classification_report_ferplus_{timestamp}.txt")
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)
print(f"Report salvato in: {report_path}")

#Confusion matrix assoluta
cm = confusion_matrix(y_true, y_pred)
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(20, 7))

sns.heatmap(cm, annot=True, fmt='d', ax=axes_cm[0],
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues', linewidths=0.5)
axes_cm[0].set_title('Confusion Matrix (conteggi assoluti)')
axes_cm[0].set_ylabel('True Label')
axes_cm[0].set_xlabel('Predicted Label')
axes_cm[0].tick_params(axis='x', rotation=45)

#Confusion matrix normalizzata (recall per classe)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=axes_cm[1],
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues', linewidths=0.5, vmin=0, vmax=1)
axes_cm[1].set_title('Confusion Matrix (normalizzata per riga — Recall)')
axes_cm[1].set_ylabel('True Label')
axes_cm[1].set_xlabel('Predicted Label')
axes_cm[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, f"confusion_matrix_ferplus_{timestamp}.png")
plt.savefig(cm_path, dpi=150)
print(f"Confusion matrix salvata in: {cm_path}")


#Salvataggio history JSON
history_combined = {
    "phase1": {
        "accuracy": history_head.history['accuracy'],
        "val_accuracy": history_head.history['val_accuracy'],
        "loss": history_head.history['loss'],
        "val_loss": history_head.history['val_loss'],
    },
    "phase2": {
        "accuracy": history_ft.history['accuracy'],
        "val_accuracy": history_ft.history['val_accuracy'],
        "loss": history_ft.history['loss'],
        "val_loss": history_ft.history['val_loss'],
    },
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

json_path = os.path.join(MODEL_DIR, f"history_mobilenetv3small_ferplus_{timestamp}.json")
with open(json_path, "w") as f:
    json.dump(history_combined, f, indent=4)
print(f"History salvata in: {json_path}")


#Grafici di Training
acc_1 = history_head.history['accuracy'] + history_ft.history['accuracy']
val_acc_1 = history_head.history['val_accuracy'] + history_ft.history['val_accuracy']
loss_1 = history_head.history['loss'] + history_ft.history['loss']
val_loss_1 = history_head.history['val_loss'] + history_ft.history['val_loss']

epochs_total = range(1, len(acc_1) + 1)
phase2_start = len(history_head.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_total, acc_1, label='Train Accuracy')
axes[0].plot(epochs_total, val_acc_1, label='Val Accuracy')
axes[0].axvline(x=phase2_start, color='gray', linestyle='--', label='Fine-tuning start')
axes[0].set_title('Accuracy - MobileNetV3Small FERPlus')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(epochs_total, loss_1, label='Train Loss')
axes[1].plot(epochs_total, val_loss_1, label='Val Loss')
axes[1].axvline(x=phase2_start, color='gray', linestyle='--', label='Fine-tuning start')
axes[1].set_title('Loss - MobileNetV3Small FERPlus')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, f"training_plot_mobilenetv3small_ferplus_{timestamp}.png")
plt.savefig(plot_path)
print(f"Grafico salvato in: {plot_path}")
