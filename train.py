"""
train.py
--------
Script di addestramento: va eseguito UNA SOLA VOLTA per preparare
il modello che poi l'app grafica userà ad ogni avvio.

Esegui con:
    python train.py
oppure, se il dataset non è nella cartella di default:
    python train.py --data percorso/al/dataset.csv
"""

import argparse
import os
import sys

# Aggiungiamo la root del progetto al path di Python,
# così possiamo importare i moduli in src/ ovunque eseguiamo lo script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_and_clean, scale_features
from src.model import build_model, save_model
from src.config import load_weights


def train(csv_path: str):
    print("=" * 50)
    print("  MUSIC RECOMMENDER — Training Pipeline")
    print("=" * 50)

    # ── Step 1: Caricamento e pulizia ──────────────
    print("\n[1/3] Caricamento e pulizia del dataset...")
    df = load_and_clean(csv_path)

    # ── Step 2: Normalizzazione delle feature ──────
    print("\n[2/3] Normalizzazione e pesatura delle feature...")
    weights = load_weights("model")
    print(f"[INFO] Pesi attivi: group_weight={weights['group_weight']}")
    X_scaled, scaler, group_columns = scale_features(
        df,
        feature_weights=weights["feature_weights"],
        group_weight=weights["group_weight"],
    )

    # ── Step 3: Addestramento e salvataggio ────────
    print("\n[3/3] Addestramento del modello KNN...")
    model = build_model(X_scaled, n_neighbors=11, metric="cosine")
    save_model(model, scaler, df, group_columns,
               feature_weights=weights["feature_weights"],
               group_weight=weights["group_weight"],
               output_dir="model")

    print("\n✅ Training completato! Ora puoi avviare l'app con: python main.py")


if __name__ == "__main__":
    # argparse gestisce gli argomenti da riga di comando.
    # Questo rende lo script flessibile: il percorso del CSV
    # può essere passato come parametro invece di essere hardcodato.
    parser = argparse.ArgumentParser(description="Addestra il modello KNN musicale.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.csv",
        help="Percorso al file CSV del dataset Spotify"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERRORE] File non trovato: {args.data}")
        print("Scarica il dataset da Kaggle e mettilo in data/dataset.csv")
        sys.exit(1)

    train(args.data)
