import numpy as np
import pandas as pd
from src.preprocessing import AUDIO_FEATURES, genre_to_group


def search_songs(query: str, df: pd.DataFrame, max_results: int = 10) -> pd.DataFrame:
    # ricerca testuale per nome o artista (non usa KNN)
    query_lower = query.lower()

    # regex=False: trattiamo la query come testo letterale, sennò '(' o '+' crashano
    mask = (
        df["track_name"].str.lower().str.contains(query_lower, na=False, regex=False) |
        df["artists"].str.lower().str.contains(query_lower, na=False, regex=False)
    )

    results = df[mask].head(max_results)
    return results


def _build_query_vector(song_row: pd.Series, scaler, group_columns,
                        feature_weights: dict, group_weight: float):
    # ricostruisce lo STESSO vettore usato in training per una singola canzone
    audio_raw = song_row[AUDIO_FEATURES].values.reshape(1, -1)
    audio_scaled = scaler.transform(audio_raw)
    weights = np.array([feature_weights[f] for f in AUDIO_FEATURES])
    audio_weighted = audio_scaled * weights

    # one-hot del macrogruppo, pesato
    group = genre_to_group(song_row.get("track_genre", ""))
    onehot = np.zeros((1, len(group_columns)), dtype=float)
    if group in group_columns:
        onehot[0, group_columns.index(group)] = 1.0
    onehot_weighted = onehot * group_weight

    return np.hstack([audio_weighted, onehot_weighted])


def get_recommendations(
    song_index: int,
    df: pd.DataFrame,
    model,
    scaler,
    group_columns,
    weights: dict,
    n_recommendations: int = 10,
) -> pd.DataFrame:
    # restituisce le N canzoni più simili a quella scelta
    query_vector = _build_query_vector(
        df.loc[song_index], scaler, group_columns,
        weights["feature_weights"], weights["group_weight"],
    )

    # ci servono per filtrare le varianti dello stesso brano
    original_name = str(df.loc[song_index, "track_name"]).lower().strip()
    original_artist = str(df.loc[song_index, "artists"]).lower().strip()

    # overfetch x5 per avere margine dopo i filtri anti-duplicato
    n_fetch = min(n_recommendations * 5 + 1, len(df))
    distances, indices = model.kneighbors(query_vector, n_neighbors=n_fetch)
    distances = distances[0]
    indices = indices[0]

    candidates = df.iloc[indices].copy()
    candidates["_distance"] = distances
    candidates["_idx"] = indices

    # escludiamo la canzone stessa
    candidates = candidates[candidates["_idx"] != song_index]

    # escludiamo varianti dello stesso titolo (remix, sped-up, deluxe, ecc.)
    candidates["_name_clean"] = candidates["track_name"].str.lower().str.strip()
    candidates["_artist_clean"] = candidates["artists"].str.lower().str.strip()

    def is_variant(row):
        cname = row["_name_clean"]
        cartist = row["_artist_clean"]
        if original_name in cname or cname in original_name:
            return True
        if cartist == original_artist:
            orig_words = set(original_name.split())
            cand_words = set(cname.split())
            if orig_words and len(orig_words & cand_words) / len(orig_words) >= 0.8:
                return True
        return False

    candidates = candidates[~candidates.apply(is_variant, axis=1)]

    # ordiniamo per distanza e prendiamo i top N
    candidates = candidates.sort_values("_distance").head(n_recommendations)

    # distanza coseno → % di similarità (0 = identico, 2 = opposto)
    candidates["similarity"] = ((1 - candidates["_distance"]) * 100).clip(0, 100).round(1)

    candidates = candidates.drop(columns=[
        "_distance", "_idx", "_name_clean", "_artist_clean"
    ], errors="ignore")

    candidates = candidates.reset_index(drop=True)
    return candidates
