"""
model.py
--------
Questo modulo costruisce e addestra il modello KNN (K-Nearest Neighbors).

Come funziona KNN per la raccomandazione?
-----------------------------------------
KNN non "impara" una formula come le reti neurali: memorizza
semplicemente tutti i punti nello spazio delle feature.
Quando gli chiedi i vicini di una canzone, calcola la distanza
tra quella canzone e TUTTE le altre, poi restituisce le K più vicine.

Analogia: immagina ogni canzone come un punto in uno spazio a 9
dimensioni (le nostre 9 feature audio). Canzoni "simili" sono
vicine in questo spazio. KNN trova i punti più vicini.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle   # per salvare il modello su disco
import os


def build_model(X_scaled: np.ndarray, n_neighbors: int = 11, metric: str = "cosine"):
    """
    Crea e addestra il modello KNN.

    Parameters
    ----------
    X_scaled : np.ndarray
        Matrice delle feature normalizzate (output di preprocessing.py).
    n_neighbors : int
        Quanti vicini considerare. Usiamo 11 perché il primo vicino
        sarà sempre la canzone stessa (distanza = 0), quindi ne
        prendiamo 10 "reali" + 1.
    metric : str
        Metrica di distanza. "cosine" misura l'angolo tra vettori
        invece della distanza euclidea pura: funziona meglio per
        feature audio perché cattura la "direzione" del suono
        indipendentemente dall'intensità assoluta.
        Altre opzioni: "euclidean", "manhattan".

    Returns
    -------
    model : NearestNeighbors
        Modello addestrato, pronto per le query.
    """

    # NearestNeighbors è la versione "non supervisionata" di KNN:
    # non prevede una label/classe, trova solo i vicini più prossimi.
    # algorithm="brute" = calcola le distanze con tutti i punti.
    # Per dataset grandi si usano "ball_tree" o "kd_tree",
    # ma con distanza coseno solo "brute" è disponibile.
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="brute"
    )

    # fit() "addestra" il modello, che nel caso di KNN significa
    # semplicemente memorizzare la matrice X_scaled.
    # Non c'è un vero calcolo qui: il lavoro pesante avviene al momento
    # della query (approccio "lazy learning").
    model.fit(X_scaled)

    print(f"[INFO] Modello KNN addestrato — k={n_neighbors}, metrica={metric}")
    return model


def save_model(model, scaler, df, group_columns,
               feature_weights=None, group_weight=None,
               output_dir: str = "model"):
    """
    Salva il modello, lo scaler, il DataFrame e la lista dei macrogruppi.

    Pickle serializza oggetti Python in file binari (.pkl).
    Questo ci permette di caricare il modello già addestrato
    senza rielaborare tutto il dataset ogni volta che avviamo l'app.

    Parameters
    ----------
    model : NearestNeighbors
        Il modello KNN addestrato.
    scaler : StandardScaler
        Lo scaler addestrato (necessario per normalizzare le query).
    df : pd.DataFrame
        Il DataFrame pulito (serve per recuperare titolo/artista dai risultati).
    group_columns : list[str]
        Lista ordinata dei macrogruppi: serve a recommender.py per ricostruire
        il one-hot del genere quando trasforma una nuova query.
    output_dir : str
        Cartella dove salvare i file.
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/knn_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{output_dir}/group_columns.pkl", "wb") as f:
        pickle.dump(group_columns, f)

    df.to_pickle(f"{output_dir}/songs_df.pkl")

    # Salviamo anche i pesi usati al training, così il recommender
    # userà sempre quelli (single source of truth per le inferenze).
    if feature_weights is not None and group_weight is not None:
        from src.config import save_weights
        save_weights(feature_weights, group_weight, model_dir=output_dir)

    print(f"[INFO] Modello salvato in '{output_dir}/'")


def load_model(model_dir: str = "model"):
    """
    Carica modello, scaler, DataFrame, lista macrogruppi e pesi attivi.

    Returns
    -------
    model, scaler, df, group_columns, weights : tuple
        weights è un dict {"feature_weights": {...}, "group_weight": float}
    """
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

    # I pesi del training (usati per costruire i vettori di query)
    weights = load_weights(model_dir)

    print("[INFO] Modello caricato dal disco")
    return model, scaler, df, group_columns, weights
