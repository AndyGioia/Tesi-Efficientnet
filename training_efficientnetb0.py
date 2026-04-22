# Fixing the seed for random number generators
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

# Import necessary modules
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import os

# Load EfficientNetB0 model and adapt for grayscale input
input_shape = (224, 224, 3)  # EfficientNetB0 expects RGB input


input_tensor = Input(shape=(224, 224, 1)) # Input per immagini grayscale
x = concatenate([input_tensor, input_tensor, input_tensor], axis=-1)

# Caricamento del modello pre-addestrato privo del top Layer e adattamento per l'input a 3 canali (grayscale duplicato su 3 canali)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
x = base_model(x, training = False)  #Usa il modello base in modalità inference 

# Global Average Pooling
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Dropout per ridurre l'overfitting
output_layer = Dense(7, activation ='softmax')(x)

# Create the model
model = Model(inputs=input_tensor, outputs=output_layer)

# Model Summary
model.summary()

# Data Augmentation

#creazione di un path relativo per il dataset, in modo da poterlo eseguire su qualsiasi computer senza dover modificare il path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "FER", "images")
MODEL_DIR = os.path.join(BASE_DIR, "models", "efficientnetb0") #al fine del confronto tra modelli, è importante salvare i modelli in una cartella dedicata all'interno del progetto, in modo da poterli confrontare facilmente e mantenere il progetto organizzato
os.makedirs(MODEL_DIR, exist_ok=True) #creazione della cartella models qualora non non dovesse esistere già

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

# Train set
train_set = datagen_train.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    shuffle=True
)

# Validation set
validation_set = datagen_validation.flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    shuffle=False
)

# Test set
test_set = datagen_test.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    shuffle=False
)

# Compute class weights dynamically
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
class_weight_dict = {k: min(v, 4.0) for k, v in class_weight_dict.items()} # Limita il peso massimo a 4.0 per evitare instabilità durante l'addestramento
print("Class weights (clipped):", class_weight_dict)

# Callbacks FASE 1
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model_b0.keras"),
    monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)


# Testing (divisione del testing in 2 fasi distinte)

#FASE 1: Training della head

base_model.trainable = False  # Congela i pesi del modello base
model.compile(
    optimizer = Adam(learning_rate = 1e-3),  # Adam con learning rate più alto per la fase di training della head
    loss = CategoricalCrossentropy(label_smoothing=0.06), #l'uso del label smoothing aiuta a prevenire l'overfitting, specialmente in un dataset relativamente piccolo e sbilanciato come il FER
    metrics = ['accuracy']
)
print("\n === FASE 1: Training HEAD ===")

start_total = time.time()
start_fase1 = time.time()

history_head = model.fit(
    train_set,
    validation_data = validation_set,
    epochs = 3, 
    callbacks= [checkpoint],
    class_weight = class_weight_dict
)

end_fase1 = time.time()
print(f"Tempo impiegato per FASE 1: {(end_fase1 - start_fase1)/60:.1f} minuti")

#FASE 2: Fine-tuning del modello completo
#Callbacks FASE 2 
checkpoint_ft = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model_b0_ft.keras"),  # file separato
    monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
early_stopping_ft = EarlyStopping(
    monitor='val_loss', patience=5, verbose=1, restore_best_weights=True
)
reduce_lr_ft = ReduceLROnPlateau(
    monitor='val_loss', factor=0.3,
    patience=2,        
    verbose=1,
    min_delta=0.001    
)
callbacks_ft = [early_stopping_ft, checkpoint_ft, reduce_lr_ft]

#Fase 2
print("\n === FASE 2: Fine-Tuning ===")

start_fase2 = time.time()

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=AdamW(learning_rate=0.00003, weight_decay=0.0001),  #AdamW con weight decay per il fine-tuning
    loss=CategoricalCrossentropy(label_smoothing=0.06),
    metrics=['accuracy']
)
history_ft = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=15,
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


class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("\n === CONFUSION MATRIX E METRICHE PER CLASSE ===")
test_set.reset()
y_pred_probs = model.predict(test_set, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_set.classes

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Salva report in txt
report_path = os.path.join(MODEL_DIR, f"classification_report_{timestamp}.txt")
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)
print(f"Report salvato in: {report_path}")

# Confusion matrix assoluta
cm = confusion_matrix(y_true, y_pred)
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', ax=axes_cm[0],
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues', linewidths=0.5)
axes_cm[0].set_title('Confusion Matrix (conteggi assoluti)')
axes_cm[0].set_ylabel('True Label')
axes_cm[0].set_xlabel('Predicted Label')
axes_cm[0].tick_params(axis='x', rotation=45)

# Confusion matrix normalizzata (recall per classe)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=axes_cm[1],
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues', linewidths=0.5, vmin=0, vmax=1)
axes_cm[1].set_title('Confusion Matrix (normalizzata per riga — Recall)')
axes_cm[1].set_ylabel('True Label')
axes_cm[1].set_xlabel('Predicted Label')
axes_cm[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=150)
print(f"Confusion matrix salvata in: {cm_path}")


# Salvataggio della history di addestramento in un JSON file, in modo da poterla analizzare successivamente


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

json_path = os.path.join(MODEL_DIR, f"history_b0_{timestamp}.json")
with open(json_path, "w") as f:
    json.dump(history_combined, f, indent=4)
print(f"History salvata in: {json_path}")

# --- Grafici ---
acc_1 = history_head.history['accuracy'] + history_ft.history['accuracy']
val_acc_1 = history_head.history['val_accuracy'] + history_ft.history['val_accuracy']
loss_1 = history_head.history['loss'] + history_ft.history['loss']
val_loss_1 = history_head.history['val_loss'] + history_ft.history['val_loss']

epochs_total = range(1, len(acc_1) + 1)
phase2_start = len(history_head.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(epochs_total, acc_1, label='Train Accuracy')
axes[0].plot(epochs_total, val_acc_1, label='Val Accuracy')
axes[0].axvline(x=phase2_start, color='gray', linestyle='--', label='Fine-tuning start')
axes[0].set_title('Accuracy - EfficientNetB0')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Loss
axes[1].plot(epochs_total, loss_1, label='Train Loss')
axes[1].plot(epochs_total, val_loss_1, label='Val Loss')
axes[1].axvline(x=phase2_start, color='gray', linestyle='--', label='Fine-tuning start')
axes[1].set_title('Loss - EfficientNetB0')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, f"training_plot_b0_{timestamp}.png")
plt.savefig(plot_path)
print(f"Grafico salvato in: {plot_path}")
