import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import os


def build_model(X_scaled: np.ndarray, n_neighbors: int = 11, metric: str = "cosine"):
    # KNN non supervisionato: trova solo i vicini, niente label
    # n_neighbors=11 perché il primo vicino è sempre la canzone stessa
    # algorithm="brute": unica opzione compatibile con metrica coseno
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="brute"
    )

    # con KNN "addestrare" significa solo memorizzare la matrice (lazy learning)
    model.fit(X_scaled)

    print(f"[INFO] Modello KNN addestrato — k={n_neighbors}, metrica={metric}")
    return model


def save_model(model, scaler, df, group_columns,
               feature_weights=None, group_weight=None,
               output_dir: str = "model"):
    # serializza modello, scaler, dataframe e colonne dei macrogruppi su disco
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/knn_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{output_dir}/group_columns.pkl", "wb") as f:
        pickle.dump(group_columns, f)

    df.to_pickle(f"{output_dir}/songs_df.pkl")

    # salviamo anche i pesi usati nel training (single source of truth)
    if feature_weights is not None and group_weight is not None:
        from src.config import save_weights
        save_weights(feature_weights, group_weight, model_dir=output_dir)

    print(f"[INFO] Modello salvato in '{output_dir}/'")


def load_model(model_dir: str = "model"):
    # ricarica tutto quello che serve all'app a runtime
    import pandas as pd
    from src.config import load_weights

    with open(f"{model_dir}/knn_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    group_columns_path = f"{model_dir}/group_columns.pkl"
    if not os.path.exists(group_columns_path):
        raise FileNotFoundError(
            "Modello obsoleto: manca 'group_columns.pkl'. "
            "Riesegui 'python train.py' per rigenerare il modello."
        )
    with open(group_columns_path, "rb") as f:
        group_columns = pickle.load(f)

    df = pd.read_pickle(f"{model_dir}/songs_df.pkl")

    weights = load_weights(model_dir)

    print("[INFO] Modello caricato dal disco")
    return model, scaler, df, group_columns, weights
