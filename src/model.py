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


def save_model(model, scaler, df, output_dir: str = "model"):
    """
    Salva il modello, lo scaler e il DataFrame su disco con pickle.

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
    output_dir : str
        Cartella dove salvare i file.
    """

    os.makedirs(output_dir, exist_ok=True)  # crea la cartella se non esiste

    with open(f"{output_dir}/knn_model.pkl", "wb") as f:
        pickle.dump(model, f)   # "wb" = write binary

    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Salviamo il DataFrame come pickle (più veloce di riscrivere il CSV)
    df.to_pickle(f"{output_dir}/songs_df.pkl")

    print(f"[INFO] Modello salvato in '{output_dir}/'")


def load_model(model_dir: str = "model"):
    """
    Carica modello, scaler e DataFrame dal disco.

    Returns
    -------
    model, scaler, df : tuple
        I tre oggetti necessari per fare raccomandazioni.
    """
    import pandas as pd

    with open(f"{model_dir}/knn_model.pkl", "rb") as f:
        model = pickle.load(f)   # "rb" = read binary

    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_pickle(f"{model_dir}/songs_df.pkl")

    print("[INFO] Modello caricato dal disco")
    return model, scaler, df
