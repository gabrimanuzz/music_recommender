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


# ─────────────────────────────────────────────
# Pesi delle feature audio (calibrazione percettiva)
# ─────────────────────────────────────────────
# Non tutte le feature sono ugualmente importanti per la "somiglianza"
# percepita tra due brani. Ad esempio liveness è spesso rumore (Spotify
# la stima male) mentre energy/valence definiscono il mood del brano.
# Dopo StandardScaler, moltiplichiamo ciascuna colonna per il suo peso.
FEATURE_WEIGHTS = {
    "danceability":     1.4,   # definisce il "groove" e il flow
    "energy":           1.5,   # driver primario del mood
    "valence":          1.4,   # asse emotivo (positivo vs malinconico)
    "acousticness":     1.3,   # distingue acustico da elettronico
    "instrumentalness": 1.0,   # utile ma rumorosa
    "tempo":            1.0,   # già correlata con energy/danceability
    "loudness":         0.7,   # più indicatore di mastering che di stile
    "speechiness":      0.6,   # utile solo per separare rap/parlato
    "liveness":         0.5,   # quasi sempre rumore di registrazione
}


# ─────────────────────────────────────────────
# Macrogruppi di genere
# ─────────────────────────────────────────────
# Il dataset Kaggle ha 114 generi distinti: molti sono varianti dello
# stesso "mondo" (es. deep-house, progressive-house, electro). Li
# raggruppiamo in 14 macrogruppi musicali. Il macrogruppo entra come
# feature one-hot nel KNN: brani dello stesso macrogruppo saranno
# automaticamente più vicini nello spazio vettoriale.
GENRE_GROUPS = {
    "pop":          ["pop", "power-pop", "synth-pop", "indie-pop", "pop-film",
                     "indie", "alternative", "emo", "romance", "british",
                     "swedish", "french", "german", "spanish"],
    "rock":         ["rock", "alt-rock", "hard-rock", "punk-rock", "psych-rock",
                     "grunge", "rock-n-roll", "rockabilly", "punk", "garage",
                     "goth", "industrial", "hardcore"],
    "metal":        ["metal", "heavy-metal", "death-metal", "black-metal",
                     "metalcore", "grindcore"],
    "electronic":   ["edm", "electronic", "electro", "techno", "house",
                     "deep-house", "chicago-house", "detroit-techno",
                     "minimal-techno", "progressive-house", "dubstep", "trance",
                     "hardstyle", "drum-and-bass", "breakbeat", "idm",
                     "trip-hop", "club", "dance", "j-dance", "disco",
                     "groove"],   # tag caotico, lo trattiamo come dance
    "hiphop":       ["hip-hop"],
    "rnb_soul":     ["soul"],   # nel dataset solo 'soul' è r&b americano pulito
    "jazz_blues":   ["jazz", "blues"],
    "folk_country": ["folk", "acoustic", "singer-songwriter", "songwriter",
                     "country", "bluegrass", "honky-tonk", "guitar"],
    "classical":    ["classical", "opera", "piano", "new-age", "ambient"],
    "latin":        ["latin", "latino", "reggaeton", "salsa", "samba", "tango",
                     "mpb", "brazil", "forro", "pagode", "sertanejo",
                     # I tag seguenti nel dataset Kaggle contengono in
                     # maggioranza musica brasiliana/latina, non occidentale:
                     "funk",     # funk carioca brasiliano
                     "gospel",   # gospel evangelico brasiliano
                     "r-n-b"],   # mix brasiliano/latino dominante
    "reggae_carib": ["reggae", "ska", "dancehall", "dub"],
    "asian":        ["j-pop", "k-pop", "anime", "j-rock", "j-idol", "cantopop",
                     "mandopop", "indian", "iranian", "turkish", "malay"],
    "world_other":  ["afrobeat", "world-music"],
    "mood_func":    ["chill", "sleep", "study", "happy", "sad", "party",
                     "kids", "children", "disney", "show-tunes", "comedy"],
}

# Lista ordinata dei macrogruppi (l'ordine fissa le colonne del one-hot).
ALL_GROUPS = sorted(GENRE_GROUPS.keys()) + ["other"]

# Peso del one-hot del macrogruppo: forte ma non dominante.
# Un mismatch di macrogruppo aggiunge ~√2 · GROUP_WEIGHT di distanza,
# paragonabile a ~2 deviazioni standard sulle feature audio pesate.
GROUP_WEIGHT = 2.5


def genre_to_group(genre: str) -> str:
    """
    Mappa un genere Spotify nel suo macrogruppo.
    Ritorna 'other' se il genere non è riconosciuto.
    """
    g = str(genre).lower().strip()
    for group, genres in GENRE_GROUPS.items():
        if g in genres:
            return group
    return "other"


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
    df["_name_lower"] = df["track_name"].str.lower().str.strip()
    df["_artist_lower"] = df["artists"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["_name_lower", "_artist_lower"])
    df = df.drop(columns=["_name_lower", "_artist_lower"])

    df = df.reset_index(drop=True)

    print(f"[INFO] Righe dopo pulizia: {len(df)}")
    return df


def scale_features(df: pd.DataFrame, feature_weights: dict = None,
                   group_weight: float = None):
    """
    Costruisce la matrice delle feature finale per il KNN:
    1. Normalizza le 9 feature audio con StandardScaler.
    2. Moltiplica ogni colonna per il suo peso (FEATURE_WEIGHTS).
    3. Aggiunge un one-hot del macrogruppo di genere, pesato GROUP_WEIGHT.

    Se feature_weights o group_weight sono None, usa i default del modulo.
    Questo permette alla GUI (tab Pesi) di passare pesi personalizzati.

    Returns
    -------
    X_final : np.ndarray
        Matrice (n_canzoni, 9 + n_macrogruppi).
    scaler : StandardScaler
        Lo scaler addestrato.
    group_columns : list[str]
        Lista ordinata dei macrogruppi (= colonne del one-hot).
    """
    if feature_weights is None:
        feature_weights = FEATURE_WEIGHTS
    if group_weight is None:
        group_weight = GROUP_WEIGHT

    # ── Step 1: scaling delle feature audio ────────────────
    X = df[AUDIO_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step 2: applicazione dei pesi colonna per colonna ──
    weights = np.array([feature_weights[f] for f in AUDIO_FEATURES])
    X_weighted = X_scaled * weights

    # ── Step 3: one-hot del macrogruppo di genere ──────────
    groups = df["track_genre"].apply(genre_to_group)
    group_to_idx = {g: i for i, g in enumerate(ALL_GROUPS)}

    X_onehot = np.zeros((len(df), len(ALL_GROUPS)), dtype=float)
    for row_idx, g in enumerate(groups):
        X_onehot[row_idx, group_to_idx[g]] = 1.0

    X_onehot_weighted = X_onehot * group_weight

    # ── Step 4: concatenazione orizzontale ─────────────────
    X_final = np.hstack([X_weighted, X_onehot_weighted])

    print(f"[INFO] Feature scalate: {X_final.shape}  "
          f"(9 audio + {len(ALL_GROUPS)} macrogruppi)")
    return X_final, scaler, ALL_GROUPS
