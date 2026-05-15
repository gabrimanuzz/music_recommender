"""
preprocessing.py
----------------
Questo modulo si occupa di caricare il dataset Spotify e prepararlo
per l'algoritmo KNN. La preparazione dei dati (preprocessing) è una
delle fasi più importanti del Machine Learning: dati "sporchi" o mal
formattati producono modelli scadenti, indipendentemente dall'algoritmo.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# Le feature audio che useremo per il confronto
# ─────────────────────────────────────────────
# Queste colonne esistono nel dataset Kaggle e rappresentano
# caratteristiche numeriche che Spotify calcola per ogni brano.
# Le abbiamo scelte perché descrivono il "suono" della canzone,
# non informazioni esterne come popolarità o anno.
AUDIO_FEATURES = [
    "danceability",      # 0.0–1.0  → quanto è adatta al ballo
    "energy",            # 0.0–1.0  → intensità e attività percepita
    "loudness",          # dB negativi → volume medio (es. -5.0)
    "speechiness",       # 0.0–1.0  → presenza di parlato
    "acousticness",      # 0.0–1.0  → probabilità che sia acustica
    "instrumentalness",  # 0.0–1.0  → assenza di voci
    "liveness",          # 0.0–1.0  → probabilità di essere live
    "valence",           # 0.0–1.0  → positività emotiva del brano
    "tempo",             # BPM      → velocità del brano
]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Carica il CSV e rimuove righe inutilizzabili.

    Parameters
    ----------
    csv_path : str
        Percorso al file dataset.csv scaricato da Kaggle.

    Returns
    -------
    pd.DataFrame
        DataFrame pulito, pronto per il preprocessing.
    """

    # pd.read_csv legge il file CSV e lo converte in un DataFrame,
    # cioè una tabella a righe e colonne simile a un foglio Excel.
    df = pd.read_csv(csv_path)

    print(f"[INFO] Righe caricate: {len(df)}")

    # dropna() rimuove le righe che contengono almeno un valore NaN
    # (Not a Number = cella vuota). KNN non funziona con valori mancanti.
    # subset=AUDIO_FEATURES → controlla solo le colonne che ci servono.
    df = df.dropna(subset=AUDIO_FEATURES)

    # Prima deduplicazione: per track_id univoco di Spotify.
    df = df.drop_duplicates(subset="track_id")

    # Seconda deduplicazione: per (track_name, artists).
    # Necessaria perché lo stesso brano può avere track_id diversi
    # se è presente in edizioni diverse (Deluxe, Remaster, ecc.).
    # Senza questo step, cercare "Blinding Lights" restituisce
    # 5 copie quasi identiche come primi risultati KNN.
    # str.lower() normalizza maiuscole prima del confronto.
    df["_name_lower"] = df["track_name"].str.lower().str.strip()
    df["_artist_lower"] = df["artists"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["_name_lower", "_artist_lower"])
    df = df.drop(columns=["_name_lower", "_artist_lower"])  # colonne temporanee

    # Resettiamo l'indice del DataFrame dopo aver rimosso righe.
    # Senza reset, gli indici avrebbero "buchi" (es. 0, 2, 5, 7...)
    # che possono causare problemi con KNN.
    df = df.reset_index(drop=True)

    print(f"[INFO] Righe dopo pulizia: {len(df)}")
    return df


def scale_features(df: pd.DataFrame):
    """
    Normalizza le feature audio con StandardScaler.

    Perché normalizzare?
    --------------------
    Le feature hanno scale molto diverse:
      - danceability va da 0 a 1
      - tempo va da 50 a 250 BPM
    Senza normalizzazione, KNN darebbe peso enorme al tempo
    solo perché i suoi valori numerici sono più grandi.
    StandardScaler trasforma ogni colonna in modo che abbia
    media = 0 e deviazione standard = 1, rendendo tutte le
    feature ugualmente influenti nel calcolo della distanza.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame pulito contenente le colonne AUDIO_FEATURES.

    Returns
    -------
    X_scaled : np.ndarray
        Matrice numpy con le feature normalizzate.
    scaler : StandardScaler
        L'oggetto scaler addestrato (serve per trasformare
        nuove canzoni prima di interrogare il modello).
    """

    # Estraiamo solo le colonne audio come matrice numpy.
    # .values converte il DataFrame in un array numpy 2D
    # di forma (n_canzoni, n_feature).
    X = df[AUDIO_FEATURES].values

    # Creiamo lo scaler e lo "addestriamo" sui nostri dati:
    # fit() calcola la media e la deviazione standard di ogni colonna.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # fit_transform è equivalente a scaler.fit(X) + scaler.transform(X)

    print(f"[INFO] Feature scalate: {X_scaled.shape}")  # (n_righe, 9)
    return X_scaled, scaler
