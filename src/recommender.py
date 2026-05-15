"""
recommender.py
--------------
Questo modulo espone la funzione principale di raccomandazione.
Fa da "ponte" tra l'interfaccia grafica e il modello KNN.
"""

import numpy as np
import pandas as pd
from src.preprocessing import AUDIO_FEATURES


def search_songs(query: str, df: pd.DataFrame, max_results: int = 10) -> pd.DataFrame:
    """
    Cerca canzoni nel dataset per nome o artista (ricerca testuale).

    Questa funzione NON usa KNN: serve solo per permettere all'utente
    di trovare la canzone di partenza nel dataset.

    Parameters
    ----------
    query : str
        Testo cercato (titolo o artista).
    df : pd.DataFrame
        Il DataFrame con tutte le canzoni.
    max_results : int
        Numero massimo di risultati da restituire.

    Returns
    -------
    pd.DataFrame
        Sottoinsieme del DataFrame con le canzoni trovate.
    """

    # str.lower() converte in minuscolo per una ricerca case-insensitive.
    # str.contains() verifica se la stringa è contenuta nella colonna.
    # na=False evita errori su celle vuote.
    query_lower = query.lower()

    # Creiamo una "maschera booleana": un array di True/False
    # che indica quali righe soddisfano la condizione.
    mask = (
        df["track_name"].str.lower().str.contains(query_lower, na=False) |
        df["artists"].str.lower().str.contains(query_lower, na=False)
    )
    # Il simbolo | è l'operatore OR applicato colonna per colonna.

    # Applichiamo la maschera al DataFrame per filtrare le righe.
    results = df[mask].head(max_results)
    return results


def get_recommendations(
    song_index: int,
    df: pd.DataFrame,
    model,
    scaler,
    n_recommendations: int = 10,
    genre_filter: bool = True
) -> pd.DataFrame:
    """
    Dato l'indice di una canzone nel DataFrame, restituisce
    le N canzoni più simili usando il modello KNN.

    Strategia anti-duplicati e anti-genere-sbagliato:
    -------------------------------------------------
    1. Chiediamo a KNN molti più vicini del necessario (x5),
       così abbiamo un pool ampio da cui filtrare.
    2. Se genre_filter=True, privilegiamo i brani dello stesso
       macrogruppo di genere della canzone originale.
    3. Escludiamo esplicitamente varianti dello stesso titolo.

    Parameters
    ----------
    song_index : int
        Indice della canzone scelta nel DataFrame.
    df : pd.DataFrame
        Il DataFrame con tutte le canzoni.
    model : NearestNeighbors
        Modello KNN addestrato.
    scaler : StandardScaler
        Scaler per normalizzare il vettore query.
    n_recommendations : int
        Quante raccomandazioni restituire.
    genre_filter : bool
        Se True, penalizza brani di generi molto lontani.

    Returns
    -------
    pd.DataFrame
        DataFrame con le canzoni raccomandate.
    """

    # ── Vettore della canzone originale ───────────────────────
    song_vector = df.loc[song_index, AUDIO_FEATURES].values.reshape(1, -1)
    song_vector_scaled = scaler.transform(song_vector)

    # Nome e artista della canzone originale (per escludere varianti).
    original_name = str(df.loc[song_index, "track_name"]).lower().strip()
    original_artist = str(df.loc[song_index, "artists"]).lower().strip()
    original_genre = str(df.loc[song_index].get("track_genre", "")).lower()

    # ── Chiediamo molti più vicini del necessario ─────────────
    # Overfetch: prendiamo n*5 vicini per avere margine di manovra
    # dopo aver filtrato duplicati e generi indesiderati.
    # Minimo 50, massimo limitato dalla dimensione del dataset.
    n_fetch = min(n_recommendations * 5 + 1, len(df))
    distances, indices = model.kneighbors(song_vector_scaled, n_neighbors=n_fetch)
    distances = distances[0]
    indices = indices[0]

    # ── Costruiamo un DataFrame intermedio con tutti i candidati ─
    candidates = df.iloc[indices].copy()
    candidates["_distance"] = distances
    candidates["_idx"] = indices

    # ── Filtro 1: escludiamo la canzone stessa ────────────────
    # Confronto per indice: escludiamo la riga con indice = song_index.
    candidates = candidates[candidates["_idx"] != song_index]

    # ── Filtro 2: escludiamo varianti dello stesso titolo ─────
    # Se il titolo contiene il nome originale (es. "Blinding Lights
    # (Remix)", "Blinding Lights - Sped Up Version"), le escludiamo.
    # Questo risolve il problema dei 5 duplicati!
    candidates["_name_clean"] = candidates["track_name"].str.lower().str.strip()
    candidates["_artist_clean"] = candidates["artists"].str.lower().str.strip()

    # Escludiamo se:
    # - il titolo del candidato CONTIENE il titolo originale, O
    # - l'artista è lo stesso E i titoli si somigliano molto
    def is_variant(row):
        cname = row["_name_clean"]
        cartist = row["_artist_clean"]
        # Stessa canzone con titolo leggermente diverso
        if original_name in cname or cname in original_name:
            return True
        # Stesso artista con titolo quasi identico (almeno 80% in comune)
        if cartist == original_artist:
            # Controlla quante parole hanno in comune
            orig_words = set(original_name.split())
            cand_words = set(cname.split())
            if orig_words and len(orig_words & cand_words) / len(orig_words) >= 0.8:
                return True
        return False

    # apply() applica la funzione is_variant a ogni riga.
    # ~ inverte la maschera booleana (teniamo le righe dove is_variant=False).
    candidates = candidates[~candidates.apply(is_variant, axis=1)]

    # ── Filtro 3 (opzionale): boost per stesso macrogruppo ────
    # Invece di escludere duramente i generi diversi, diamo un
    # "bonus" alle canzoni dello stesso macrogruppo riducendo
    # artificialmente la loro distanza del 20%.
    # Questo preserva buoni risultati da generi affini (es. pop/dance)
    # ma penalizza canzoni giapponesi, classica, ecc.
    if genre_filter and "track_genre" in df.columns:
        # Macrogruppi di generi: canzoni dello stesso gruppo ricevono bonus.
        GENRE_GROUPS = {
            "pop":       ["pop", "dance", "power-pop", "synth-pop", "electropop", "indie-pop"],
            "rock":      ["rock", "alt-rock", "hard-rock", "punk-rock", "psych-rock", "grunge"],
            "electronic":["edm", "electronic", "techno", "house", "dubstep", "trance", "disco"],
            "hiphop":    ["hip-hop", "rap", "trap"],
            "rnb":       ["r-n-b", "soul", "funk"],
            "latin":     ["latin", "reggaeton", "salsa", "samba", "mpb"],
            "metal":     ["metal", "heavy-metal", "death-metal", "black-metal"],
            "folk":      ["folk", "acoustic", "singer-songwriter", "country"],
            "jazz":      ["jazz", "blues"],
            "classical": ["classical", "opera"],
            "asian":     ["j-pop", "k-pop", "anime", "j-rock", "j-idol"],
        }

        # Troviamo il macrogruppo della canzone originale
        original_group = None
        for group, genres in GENRE_GROUPS.items():
            if any(g in original_genre for g in genres):
                original_group = group
                break

        if original_group is not None:
            group_genres = GENRE_GROUPS[original_group]
            def same_group(genre_str):
                g = str(genre_str).lower()
                return any(gg in g for gg in group_genres)

            in_group = candidates["track_genre"].apply(same_group)

            # Riduciamo la distanza del 20% per i brani dello stesso gruppo.
            # Questo li fa salire in cima alla classifica dei vicini.
            candidates.loc[in_group, "_distance"] *= 0.80

    # ── Ordiniamo per distanza (aggiornata) e prendiamo i top N ─
    candidates = candidates.sort_values("_distance").head(n_recommendations)

    # ── Costruiamo il risultato finale ────────────────────────
    # Distanza coseno: 0 = identico, 2 = opposto.
    # Similarità = (1 - distanza) * 100, clampata tra 0 e 100.
    candidates["similarity"] = ((1 - candidates["_distance"]) * 100).clip(0, 100).round(1)

    # Rimuoviamo le colonne temporanee usate solo internamente.
    candidates = candidates.drop(columns=[
        "_distance", "_idx", "_name_clean", "_artist_clean"
    ], errors="ignore")

    candidates = candidates.reset_index(drop=True)
    return candidates
