import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# feature audio numeriche che descrivono il "suono" della canzone
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


# pesi delle feature: dopo lo scaling, ogni colonna viene moltiplicata per il suo peso
FEATURE_WEIGHTS = {
    "danceability":     1.4,
    "energy":           1.5,
    "valence":          1.4,
    "acousticness":     1.3,
    "instrumentalness": 1.0,
    "tempo":            1.0,
    "loudness":         0.7,
    "speechiness":      0.6,
    "liveness":         0.5,
}


# i 114 generi del dataset raggruppati in macrogruppi (entrano come one-hot nel KNN)
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
                     "groove"],
    "hiphop":       ["hip-hop"],
    "rnb_soul":     ["soul"],
    "jazz_blues":   ["jazz", "blues"],
    "folk_country": ["folk", "acoustic", "singer-songwriter", "songwriter",
                     "country", "bluegrass", "honky-tonk", "guitar"],
    "classical":    ["classical", "opera", "piano", "new-age", "ambient"],
    "latin":        ["latin", "latino", "reggaeton", "salsa", "samba", "tango",
                     "mpb", "brazil", "forro", "pagode", "sertanejo",
                     # nel dataset Kaggle questi tag sono in maggioranza brasiliani/latini
                     "funk",
                     "gospel",
                     "r-n-b"],
    "reggae_carib": ["reggae", "ska", "dancehall", "dub"],
    "asian":        ["j-pop", "k-pop", "anime", "j-rock", "j-idol", "cantopop",
                     "mandopop", "indian", "iranian", "turkish", "malay"],
    "world_other":  ["afrobeat", "world-music"],
    "mood_func":    ["chill", "sleep", "study", "happy", "sad", "party",
                     "kids", "children", "disney", "show-tunes", "comedy"],
}

# l'ordine fissa le colonne del one-hot
ALL_GROUPS = sorted(GENRE_GROUPS.keys()) + ["other"]

# peso del one-hot del macrogruppo (forte ma non dominante)
GROUP_WEIGHT = 2.5


def genre_to_group(genre: str) -> str:
    # mappa un genere Spotify nel suo macrogruppo, 'other' se non riconosciuto
    g = str(genre).lower().strip()
    for group, genres in GENRE_GROUPS.items():
        if g in genres:
            return group
    return "other"


def load_and_clean(csv_path: str) -> pd.DataFrame:
    # carica il CSV e rimuove righe inutilizzabili
    df = pd.read_csv(csv_path)
    print(f"[INFO] Righe caricate: {len(df)}")

    # KNN non funziona con valori mancanti
    df = df.dropna(subset=AUDIO_FEATURES)

    # dedup per track_id univoco
    df = df.drop_duplicates(subset="track_id")

    # dedup per (nome, artista): stesso brano con track_id diversi (Deluxe, Remaster, ecc.)
    df["_name_lower"] = df["track_name"].str.lower().str.strip()
    df["_artist_lower"] = df["artists"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["_name_lower", "_artist_lower"])
    df = df.drop(columns=["_name_lower", "_artist_lower"])

    df = df.reset_index(drop=True)
    print(f"[INFO] Righe dopo pulizia: {len(df)}")
    return df


def scale_features(df: pd.DataFrame, feature_weights: dict = None,
                   group_weight: float = None):
    # costruisce la matrice finale per il KNN: 9 audio scalate+pesate + one-hot pesato
    if feature_weights is None:
        feature_weights = FEATURE_WEIGHTS
    if group_weight is None:
        group_weight = GROUP_WEIGHT

    # scaling delle feature audio
    X = df[AUDIO_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # applicazione dei pesi colonna per colonna
    weights = np.array([feature_weights[f] for f in AUDIO_FEATURES])
    X_weighted = X_scaled * weights

    # one-hot del macrogruppo di genere
    groups = df["track_genre"].apply(genre_to_group)
    group_to_idx = {g: i for i, g in enumerate(ALL_GROUPS)}

    X_onehot = np.zeros((len(df), len(ALL_GROUPS)), dtype=float)
    for row_idx, g in enumerate(groups):
        X_onehot[row_idx, group_to_idx[g]] = 1.0

    X_onehot_weighted = X_onehot * group_weight

    X_final = np.hstack([X_weighted, X_onehot_weighted])

    print(f"[INFO] Feature scalate: {X_final.shape}  "
          f"(9 audio + {len(ALL_GROUPS)} macrogruppi)")
    return X_final, scaler, ALL_GROUPS
