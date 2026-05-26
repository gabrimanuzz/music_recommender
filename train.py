import argparse
import os
import sys

# aggiungiamo la root del progetto al path così src/ è importabile da ovunque
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean, scale_features
from src.model import build_model, save_model
from src.config import load_weights


def train(csv_path: str):
    print("=" * 50)
    print("  MUSIC RECOMMENDER — Training Pipeline")
    print("=" * 50)

    # caricamento e pulizia
    print("\n[1/3] Caricamento e pulizia del dataset...")
    df = load_and_clean(csv_path)

    # normalizzazione e pesatura
    print("\n[2/3] Normalizzazione e pesatura delle feature...")
    weights = load_weights("model")
    print(f"[INFO] Pesi attivi: group_weight={weights['group_weight']}")
    X_scaled, scaler, group_columns = scale_features(
        df,
        feature_weights=weights["feature_weights"],
        group_weight=weights["group_weight"],
    )

    # addestramento e salvataggio
    print("\n[3/3] Addestramento del modello KNN...")
    model = build_model(X_scaled, n_neighbors=11, metric="cosine")
    save_model(model, scaler, df, group_columns,
               feature_weights=weights["feature_weights"],
               group_weight=weights["group_weight"],
               output_dir="model")

    print("\n✅ Training completato! Ora puoi avviare l'app con: python main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestra il modello KNN musicale.")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "dataset.csv"),
        help="Percorso al file CSV del dataset Spotify"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERRORE] File non trovato: {args.data}")
        print("Scarica il dataset da Kaggle e mettilo in data/dataset.csv")
        sys.exit(1)

    # ci spostiamo nella root del progetto così model/ viene scritto al posto giusto
    os.chdir(PROJECT_ROOT)

    train(args.data)
