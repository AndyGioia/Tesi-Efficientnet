import os
import shutil
import random
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
SEED         = 42
TRAIN_RATIO  = 0.875   # 87.5% come nel paper
CLASSES      = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))

DATA_DIR  = "/mnt/i/TESI/Tesi-Efficientnet/FER/images"
NEW_TRAIN = "/mnt/i/TESI/Tesi-Efficientnet/FER/images/train_new"
NEW_VAL   = "/mnt/i/TESI/Tesi-Efficientnet/FER/images/validation_new"
SRC_TRAIN = "/mnt/i/TESI/Tesi-Efficientnet/FER/images/train"
SRC_VAL   = "/mnt/i/TESI/Tesi-Efficientnet/FER/images/validation"
# ────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

print("=== RESPLIT FER2013: 87.5% train / 12.5% val ===\n")

for cls in CLASSES:
    # Raccogli tutti i file da train/ e validation/ per questa classe
    src_files = []
    for src_dir in [SRC_TRAIN, SRC_VAL]:
        cls_dir = os.path.join(src_dir, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        src_files.extend(files)
    print(f"{cls}: trovati {len(src_files)} file")

    random.shuffle(src_files)

    split_idx   = int(len(src_files) * TRAIN_RATIO)
    train_files = src_files[:split_idx]
    val_files   = src_files[split_idx:]

    # Crea cartelle destinazione
    os.makedirs(os.path.join(NEW_TRAIN, cls), exist_ok=True)
    os.makedirs(os.path.join(NEW_VAL,   cls), exist_ok=True)

    # Copia (non muove, così le originali restano intatte)
    for f in train_files:
        shutil.copy2(f, os.path.join(NEW_TRAIN, cls, os.path.basename(f)))
    for f in val_files:
        shutil.copy2(f, os.path.join(NEW_VAL, cls, os.path.basename(f)))

    print(f"  {cls:10s} → train: {len(train_files):5d} | val: {len(val_files):4d} | totale: {len(src_files):5d}")

print("\n✅ Resplit completato.")
print(f"   Nuove cartelle: train_new/ e validation_new/")
print(f"   Le originali train/ e validation/ sono intatte.")
print(f"\n  Quando il resplit ti sembra ok, esegui il rename:")
print(f"   mv train train_old && mv validation validation_old")
print(f"   mv train_new train && mv validation_new validation")