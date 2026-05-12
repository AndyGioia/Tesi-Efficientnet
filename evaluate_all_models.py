"""
Valutazione comparativa di tutti e 6 i modelli addestrati
(EfficientNetB0, MobileNetV3Small, EfficientNetV2B2)
su FER2013 e FERPlus, con metriche complete e grafici comparativi.
"""
import numpy as np
import os
import time
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ─── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output separato per non sovrascrivere i risultati FER2013 precedenti
OUT_DIR = os.path.join(BASE_DIR, "evaluation_results_all_6")
os.makedirs(OUT_DIR, exist_ok=True)

# I due dataset hanno strutture di cartelle distinte: path separati
DATA_DIR_FER     = os.path.join(BASE_DIR, "FER",     "images")
DATA_DIR_FERPLUS = os.path.join(BASE_DIR, "FERPlus")

# FER2013 ha 7 classi, FERPlus ne ha 8 (aggiunge "contempt"):
# definirle separatamente evita che flow_from_directory carichi classi sbagliate
CLASSES_FER     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASSES_FERPLUS = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BATCH_SIZE = 32
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")


# ─── Lista modelli ─────────────────────────────────────────────────────────────
# Ogni entry include "dataset", "data_dir" e "classes" così il loop di valutazione
# non ha bisogno di logica condizionale: legge tutto da cfg
MODELS = [
    # ── FER2013 ──────────────────────────────────────────────────────────────
    {
        "name":     "EfficientNetV2B2",
        "dataset":  "FER2013",
        "path":     "/mnt/i/TESI/Tesi-Efficientnet/models/efficientnetb2v2/best_model_v2b2_ft.keras",
        "data_dir": DATA_DIR_FER,
        "classes":  CLASSES_FER,
        "img_size": 260,
        "color":    "#01696f"
    },
    {
        "name":     "EfficientNetB0",
        "dataset":  "FER2013",
        "path":     "/mnt/i/TESI/Tesi-Efficientnet/models/efficientnetb0/best_model_b0_ft.keras",
        "data_dir": DATA_DIR_FER,
        "classes":  CLASSES_FER,
        "img_size": 224,
        "color":    "#fdb462"
    },
    {
        "name":     "MobileNetV3Small",
        "dataset":  "FER2013",
        "path":     "/mnt/i/TESI/Tesi-Efficientnet/models/mobilenetv3small/best_model_mobilenetv3small_ft.keras",
        "data_dir": DATA_DIR_FER,
        "classes":  CLASSES_FER,
        "img_size": 224,
        "color":    "#e377c2"
    },
    # ── FERPlus ───────────────────────────────────────────────────────────────
    {
        "name":     "EfficientNetV2B2",
        "dataset":  "FERPlus",
        "path":     "/mnt/i/TESI/Tesi-Efficientnet/models/efficientnetb2v2_ferplus/best_model_v2b2_ferplus_ft.keras",
        "data_dir": DATA_DIR_FERPLUS,
        "classes":  CLASSES_FERPLUS,
        "img_size": 260,
        "color":    "#01696f"
    },
    {
        "name":     "EfficientNetB0",
        "dataset":  "FERPlus",
        "path":     "/mnt/i/TESI/Tesi-Efficientnet/models/efficientnetb0_ferplus/best_model_b0_ferplus_ft.keras",
        "data_dir": DATA_DIR_FERPLUS,
        "classes":  CLASSES_FERPLUS,
        "img_size": 224,
        "color":    "#fdb462"
    },
    {
        "name":     "MobileNetV3Small",
        "dataset":  "FERPlus",
        "path":     "/mnt/i/TESI/Tesi-Efficientnet/models/mobilenetv3small_ferplus/best_model_mobilenetv3small_ferplus_ft.keras",
        "data_dir": DATA_DIR_FERPLUS,
        "classes":  CLASSES_FERPLUS,
        "img_size": 224,
        "color":    "#e377c2"
    },
]


# ─── Funzioni di supporto ──────────────────────────────────────────────────────

def measure_latency_cpu(model, sample, n_warmup=5, n_runs=100):
    # Misura la latenza CPU su singola immagine, con warmup e media su più esecuzioni
    with tf.device('/CPU:0'):
        for _ in range(n_warmup):
            model(sample, training=False)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(sample, training=False)
            latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms":   round(float(np.mean(latencies)), 1),
        "median_ms": round(float(np.median(latencies)), 1),
        "p95_ms":    round(float(np.percentile(latencies, 95)), 1),
        "min_ms":    round(float(np.min(latencies)), 1),
        "max_ms":    round(float(np.max(latencies)), 1)
    }


def compute_all_metrics(y_true, y_pred, y_probs, classes):
    # Calcola tutte le metriche standard di valutazione per classificazione multi-classe.
    # Riceve 'classes' come parametro per supportare sia 7 (FER2013) che 8 (FERPlus) classi
    n_classes   = len(classes)
    report_dict = classification_report(
        y_true, y_pred, target_names=classes, digits=4, output_dict=True
    )
    y_true_bin   = label_binarize(y_true, classes=list(range(n_classes)))
    auc_macro    = roc_auc_score(y_true_bin, y_probs, multi_class='ovr', average='macro')
    auc_weighted = roc_auc_score(y_true_bin, y_probs, multi_class='ovr', average='weighted')
    auc_per_class = {
        cls: round(roc_auc_score(y_true_bin[:, i], y_probs[:, i]), 4)
        for i, cls in enumerate(classes)
    }

    ce_loss  = log_loss(y_true, y_probs)
    kappa    = cohen_kappa_score(y_true, y_pred)
    mcc      = matthews_corrcoef(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    top2_acc = top_k_accuracy_score(y_true, y_probs, k=2)
    top3_acc = top_k_accuracy_score(y_true, y_probs, k=3)

    per_class = {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1-score":  round(report_dict[cls]["f1-score"],  4),
            "support":   report_dict[cls]["support"],
            "auc":       auc_per_class[cls]
        }
        for cls in classes
    }

    return {
        "accuracy":           round(report_dict["accuracy"], 4),
        "balanced_accuracy":  round(bal_acc,     4),
        "top2_accuracy":      round(top2_acc,    4),
        "top3_accuracy":      round(top3_acc,    4),
        "macro_precision":    round(report_dict["macro avg"]["precision"],    4),
        "macro_recall":       round(report_dict["macro avg"]["recall"],       4),
        "macro_f1":           round(report_dict["macro avg"]["f1-score"],     4),
        "weighted_precision": round(report_dict["weighted avg"]["precision"], 4),
        "weighted_recall":    round(report_dict["weighted avg"]["recall"],    4),
        "weighted_f1":        round(report_dict["weighted avg"]["f1-score"],  4),
        "auc_macro":          round(auc_macro,    4),
        "auc_weighted":       round(auc_weighted, 4),
        "cross_entropy_loss": round(ce_loss, 4),
        "cohen_kappa":        round(kappa,   4),
        "matthews_mcc":       round(mcc,     4),
        "per_class":          per_class,
    }


def plot_confusion_matrix(cm, classes, model_name, acc, out_path):
    # Salva la confusion matrix assoluta e normalizzata.
    # Riceve 'classes' come parametro per adattarsi a 7 o 8 classi senza modifiche
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Conteggi assoluti", "Normalizzata (Recall per riga)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, ax=ax,
                    xticklabels=classes,
                    yticklabels=classes,
                    cmap="Blues", linewidths=0.4,
                    vmin=0, vmax=None if fmt == "d" else 1)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"Confusion Matrix — {model_name} | Accuracy: {acc*100:.2f}%",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_f1_comparision(results, classes, out_path, title_suffix=""):
    # Riceve 'classes' come parametro perché FER2013 e FERPlus hanno liste diverse.
    # La larghezza delle barre è adattiva: con 3 modelli per grafico il valore
    # originale 0.25 funziona bene, ma rende il codice più robusto al numero di modelli
    x     = np.arange(len(classes))
    width = max(0.15, 0.7 / len(results))
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, res in enumerate(results):
        f1s  = [res["metrics"]["per_class"][c]["f1-score"] for c in classes]
        bars = ax.bar(
            x + i * width, f1s, width,
            label=f"{res['name']} ({res['dataset']})",
            color=res["color"], alpha=0.85, edgecolor="white"
        )
        for bar, val in zip(bars, f1s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, color=res["color"]
            )

    # I tick sono centrati rispetto al gruppo di barre
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"F1-Score per classe — {title_suffix}")
    ax.legend(loc="lower right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_radar_chart(results, out_path):
    # Radar chart delle metriche globali. Invariato rispetto all'originale:
    # opera solo su chiavi del dizionario metrics, indipendente dal numero di classi
    labels = ["Accuracy", "Bal. Acc", "Macro F1", "Top-2 Acc", "AUC\nMacro", "Cohen κ"]
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="#888888")
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(pad=30)
    ax.grid(color="#cccccc", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.spines["polar"].set_color("#cccccc")

    for res in results:
        m      = res["metrics"]
        values = [
            m["accuracy"],
            m["balanced_accuracy"],
            m["macro_f1"],
            m["top2_accuracy"],
            m["auc_macro"],
            (m["cohen_kappa"] + 1) / 2,  # Normalizza Cohen κ da [-1,1] a [0,1]
        ]
        values += values[:1]
        label = f"{res['name']} ({res['dataset']})"
        ax.plot(angles, values, color=res["color"], linewidth=2.8, zorder=3, label=label)
        ax.fill(angles, values, color=res["color"], alpha=0.08, zorder=2)

    ax.set_thetagrids(np.degrees(angles[:-1]), [""] * n)
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 1.22, label,
                ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#333333",
                transform=ax.get_xaxis_transform())
    ax.tick_params(axis='x', pad=0)
    ax.set_title("Radar — Metriche Globali", fontsize=14,
                 fontweight="bold", color="#222222", pad=30)

    handles = [mpatches.Patch(color=r["color"], label=f"{r['name']} ({r['dataset']})") for r in results]
    ax.legend(handles=handles,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=10,
              frameon=True, framealpha=0.9, edgecolor="#dddddd")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_latency_comparision(results, out_path):
    # Bar chart delle latenze CPU. Invariato rispetto all'originale:
    # il titolo del grafico mostrerà il dataset corretto grazie a title_suffix
    names  = [r["name"] for r in results]
    means  = [r["latency"]["mean_ms"] for r in results]
    p95s   = [r["latency"]["p95_ms"]  for r in results]
    colors = [r["color"]              for r in results]
    errs   = [p - m for p, m in zip(p95s, means)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x    = range(len(names))
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white",
                  yerr=errs, capsize=6,
                  error_kw=dict(ecolor="#555555", lw=1.5, zorder=5))

    for i, (val, err) in enumerate(zip(means, errs)):
        y_top = val + err
        ax.text(i, y_top + 10, f"{val:.1f} ms",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", zorder=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Latenza CPU — ms (batch = 1)")
    ax.set_title("Latenza di inferenza su CPU — confronto modelli\n"
                 "Barra = media, error bar = p95")
    ax.set_ylim(0, max(p95s) * 1.22)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto(results, out_path, title="Curva Pareto: Accuracy vs Dimensione"):
    # Il range Y era hardcoded per FER2013 (0.630–0.715).
    # Con FERPlus i valori salgono a ~0.76-0.82: range adattivo calcolato dai dati.
    # Marker diverso (● FER2013, ▲ FERPlus) per distinguere i due dataset in un unico grafico
    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        marker = "o" if res["dataset"] == "FER2013" else "^"
        ax.scatter(
            res["model_size_mb"], res["metrics"]["accuracy"],
            color=res["color"], s=160, zorder=5,
            marker=marker, edgecolors="white", linewidths=2
        )
        ax.annotate(
            f"{res['name']}\n({res['dataset']}) {res['metrics']['accuracy']*100:.2f}%"
            f" · {res['model_size_mb']:.1f} MB",
            xy=(res["model_size_mb"], res["metrics"]["accuracy"]),
            xytext=(res["model_size_mb"] + 3, res["metrics"]["accuracy"] + 0.003),
            fontsize=8, color=res["color"],
            arrowprops=dict(arrowstyle="-", color=res["color"], lw=0.8),
        )

    ax.set_xlabel("Dimensione modello su disco (MB)")
    ax.set_ylabel("Test Accuracy")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.set_title(title)
    ax.grid(alpha=0.2)

    all_accs = [r["metrics"]["accuracy"] for r in results]
    ax.set_ylim(min(all_accs) - 0.02, max(all_accs) + 0.02)

    # Legenda manuale per marker: non basta il colore, serve distinguere il dataset
    legend_handles = [
        mpatches.Patch(color=r["color"], label=f"{r['name']} ({r['dataset']})")
        for r in results
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_comparison_report(results, out_path):
    # Il report originale aveva un'unica sezione piatta con CLASSES globali.
    # Con 6 modelli su due dataset servono sezioni separate per non mischiare
    # metriche per classe che hanno nomi e conteggi diversi (7 vs 8 classi)
    lines = []
    sep   = "=" * 80
    lines += [sep,
              " REPORT COMPARATIVO — TUTTI E 6 I MODELLI (FER2013 + FERPlus)",
              f" Generato: {TIMESTAMP}", sep]

    for dataset_name, dataset_classes in [("FER2013", CLASSES_FER), ("FERPlus", CLASSES_FERPLUS)]:
        subset = [r for r in results if r["dataset"] == dataset_name]
        lines += [f"\n{'─'*80}", f" DATASET: {dataset_name}", f"{'─'*80}"]

        header = f"{'Metrica':<26}" + "".join(f"{r['name']:>22}" for r in subset)
        lines += [header, "-" * (26 + 22 * len(subset))]

        global_rows = [
            ("Accuracy (%)",        lambda m: f"{m['accuracy']*100:.2f}"),
            ("Balanced Accuracy",   lambda m: f"{m['balanced_accuracy']:.4f}"),
            ("Top-2 Accuracy",      lambda m: f"{m['top2_accuracy']:.4f}"),
            ("Top-3 Accuracy",      lambda m: f"{m['top3_accuracy']:.4f}"),
            ("Macro Precision",     lambda m: f"{m['macro_precision']:.4f}"),
            ("Macro Recall",        lambda m: f"{m['macro_recall']:.4f}"),
            ("Macro F1",            lambda m: f"{m['macro_f1']:.4f}"),
            ("Weighted F1",         lambda m: f"{m['weighted_f1']:.4f}"),
            ("AUC-ROC Macro",       lambda m: f"{m['auc_macro']:.4f}"),
            ("AUC-ROC Weighted",    lambda m: f"{m['auc_weighted']:.4f}"),
            ("Cross-Entropy Loss",  lambda m: f"{m['cross_entropy_loss']:.4f}"),
            ("Cohen Kappa",         lambda m: f"{m['cohen_kappa']:.4f}"),
            ("Matthews MCC",        lambda m: f"{m['matthews_mcc']:.4f}"),
        ]
        for label, fn in global_rows:
            lines.append(
                f"{label:<26}" + "".join(f"{fn(r['metrics']):>22}" for r in subset)
            )

        lines += [f"\n F1-Score per classe — {dataset_name}"]
        hdr2 = f"{'Classe':<14}" + "".join(f"{r['name']:>22}" for r in subset)
        lines += [hdr2, "-" * (14 + 22 * len(subset))]
        for cls in dataset_classes:
            lines.append(
                f"{cls:<14}" +
                "".join(f"{r['metrics']['per_class'][cls]['f1-score']:>22.4f}" for r in subset)
            )

        lines += [f"\n AUC-ROC per classe — {dataset_name}"]
        lines += [hdr2, "-" * (14 + 22 * len(subset))]
        for cls in dataset_classes:
            lines.append(
                f"{cls:<14}" +
                "".join(f"{r['metrics']['per_class'][cls]['auc']:>22.4f}" for r in subset)
            )

        lines += [f"\n Hardware e deployment — {dataset_name}"]
        lines += [header, "-" * (26 + 22 * len(subset))]
        hw_rows = [
            ("Dimensione (MB)",       lambda r: f"{r['model_size_mb']:.2f}"),
            ("Parametri totali",      lambda r: f"{r['total_params']:,}"),
            ("Latenza CPU mean (ms)", lambda r: f"{r['latency']['mean_ms']:.1f}"),
            ("Latenza CPU p95 (ms)",  lambda r: f"{r['latency']['p95_ms']:.1f}"),
        ]
        for label, fn in hw_rows:
            lines.append(
                f"{label:<26}" + "".join(f"{fn(r):>22}" for r in subset)
            )

    lines.append("\n" + sep)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─── Loop di valutazione ──────────────────────────────────────────────────────
results = []
for cfg in MODELS:
    name     = cfg["name"]
    dataset  = cfg["dataset"]
    path     = cfg["path"]
    img_size = cfg["img_size"]
    classes  = cfg["classes"]   # Letto da cfg: evita if/else dataset nel loop
    data_dir = cfg["data_dir"]  # Letto da cfg: evita if/else dataset nel loop
    label    = f"{name} ({dataset})"

    print(f"\n{'='*60}")
    print(f" VALUTAZIONE: {label}")
    print(f"{'='*60}\n")

    model = load_model(path)
    print(f"Modello caricato da: {path}")

    datagen_test = ImageDataGenerator()
    test_set = datagen_test.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes,   # Passato esplicitamente per garantire ordine corretto delle classi
        shuffle=False
    )

    print("[1/4] Valutazione Keras — Test Loss e Test Accuracy...")
    test_loss, test_accuracy = model.evaluate(test_set, verbose=1)

    print("[2/4] Predizioni sul test set...")
    test_set.reset()
    y_probs = model.predict(test_set, verbose=1)
    y_pred  = np.argmax(y_probs, axis=1)
    y_true  = test_set.classes

    print("[3/4] Calcolo metriche...")
    metrics = compute_all_metrics(y_true, y_pred, y_probs, classes)

    print("[4/4] Misura latenza CPU...")
    test_set.reset()
    sample  = next(iter(test_set))[0][:1]
    latency = measure_latency_cpu(model, sample)

    model_size_mb = os.path.getsize(path) / (1024 * 1024)
    total_params  = model.count_params()

    # Nome file include dataset per non sovrascrivere output FER2013 con FERPlus
    safe_name = f"{name}_{dataset}"
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm, classes, label, test_accuracy,
        os.path.join(OUT_DIR, f"confusion_matrix_{safe_name}_{TIMESTAMP}.png")
    )

    report_str = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(os.path.join(OUT_DIR, f"classification_report_{safe_name}_{TIMESTAMP}.txt"), "w") as f:
        f.write(f"Modello: {label}\nData: {TIMESTAMP}\n")
        f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\nTest Loss: {test_loss:.4f}\n\n")
        f.write(report_str)

    results.append({
        "name":          name,
        "dataset":       dataset,   # Necessario per filtrare i risultati nei grafici per dataset
        "path":          path,
        "color":         cfg["color"],
        "metrics":       metrics,
        "test_loss":     round(float(test_loss), 4),
        "model_size_mb": round(model_size_mb, 2),
        "total_params":  int(total_params),
        "latency":       latency
    })

    print(f"\n Accuracy:   {test_accuracy*100:.2f}%")
    print(f" AUC Macro:  {metrics['auc_macro']:.4f}")
    print(f" Cohen κ:    {metrics['cohen_kappa']:.4f}")
    print(f" MCC:        {metrics['matthews_mcc']:.4f}")
    print(f" Latenza:    {latency['mean_ms']:.1f} ms (mean) | {latency['p95_ms']:.1f} ms (p95)")

    del model
    tf.keras.backend.clear_session()


# ─── Grafici comparativi ──────────────────────────────────────────────────────
print("\n Generazione grafici comparativi...")

# Un grafico F1, radar e latenza per ciascun dataset separatamente,
# così le classi (7 vs 8) non si mescolano e i grafici restano leggibili
for dataset_name, dataset_classes in [("FER2013", CLASSES_FER), ("FERPlus", CLASSES_FERPLUS)]:
    subset = [r for r in results if r["dataset"] == dataset_name]
    plot_per_class_f1_comparision(
        subset, dataset_classes,
        os.path.join(OUT_DIR, f"f1_comparision_{dataset_name}_{TIMESTAMP}.png"),
        title_suffix=dataset_name
    )
    plot_radar_chart(
        subset,
        os.path.join(OUT_DIR, f"radar_{dataset_name}_{TIMESTAMP}.png")
    )
    plot_latency_comparision(
        subset,
        os.path.join(OUT_DIR, f"latency_{dataset_name}_{TIMESTAMP}.png")
    )

# Pareto e radar globali su tutti e 6: utili per la tesi per mostrare
# il trade-off accuracy/dimensione tra dataset e architetture in un colpo solo
plot_pareto(
    results,
    os.path.join(OUT_DIR, f"pareto_all6_{TIMESTAMP}.png"),
    title="Curva Pareto: Accuracy vs Dimensione — ● FER2013  ▲ FERPlus"
)
plot_radar_chart(
    results,
    os.path.join(OUT_DIR, f"radar_all6_{TIMESTAMP}.png")
)

# Report testuale e JSON
save_comparison_report(
    results,
    os.path.join(OUT_DIR, f"comparison_report_all6_{TIMESTAMP}.txt")
)
with open(os.path.join(OUT_DIR, f"comparison_results_all6_{TIMESTAMP}.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)


# ─── Riepilogo finale su console ──────────────────────────────────────────────
print(f"\n{'='*78}")
print(f" {'Modello':<20} {'Dataset':<10} {'Acc':>7} {'AUC':>7} {'κ':>7} {'MCC':>7} {'wF1':>7} {'MB':>6} {'ms':>5}")
print("─" * 78)
for r in results:
    m = r["metrics"]
    print(
        f" {r['name']:<20} {r['dataset']:<10} "
        f"{m['accuracy']*100:>6.2f}% "
        f"{m['auc_macro']:>7.4f} "
        f"{m['cohen_kappa']:>7.4f} "
        f"{m['matthews_mcc']:>7.4f} "
        f"{m['weighted_f1']:>7.4f} "
        f"{r['model_size_mb']:>5.1f}MB "
        f"{r['latency']['mean_ms']:>4.1f}ms"
    )
print(f"{'='*78}")
print(f"\n Output salvati in: {OUT_DIR}")