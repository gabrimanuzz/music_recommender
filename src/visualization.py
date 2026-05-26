import numpy as np
from sklearn.decomposition import PCA
from src.preprocessing import AUDIO_FEATURES, genre_to_group


def build_feature_matrix(df, scaler, feature_weights: dict,
                         group_weight: float, group_columns):
    # ricostruisce la matrice di feature usata dal KNN (non la salviamo, ricalcolo veloce)
    audio_scaled = scaler.transform(df[AUDIO_FEATURES].values)
    weights = np.array([feature_weights[f] for f in AUDIO_FEATURES])
    X_audio = audio_scaled * weights

    # one-hot del macrogruppo
    groups = df["track_genre"].apply(genre_to_group)
    group_to_idx = {g: i for i, g in enumerate(group_columns)}
    X_onehot = np.zeros((len(df), len(group_columns)), dtype=float)
    for row_idx, g in enumerate(groups):
        X_onehot[row_idx, group_to_idx.get(g, -1)] = 1.0
    X_onehot *= group_weight

    return np.hstack([X_audio, X_onehot])


def compute_pca_sample(df, scaler, feature_weights, group_weight,
                       group_columns, n_sample: int = 3000, seed: int = 42):
    # PCA 2D su un campione casuale di brani
    n_sample = min(n_sample, len(df))
    sample_df = df.sample(n=n_sample, random_state=seed)

    X = build_feature_matrix(sample_df, scaler, feature_weights,
                             group_weight, group_columns)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)

    groups = sample_df["track_genre"].apply(genre_to_group).values
    return sample_df, coords, groups, pca


def project_point(song_row, scaler, feature_weights, group_weight,
                  group_columns, pca):
    # proietta un singolo brano nello spazio PCA già fittato
    audio_scaled = scaler.transform(song_row[AUDIO_FEATURES].values.reshape(1, -1))
    weights = np.array([feature_weights[f] for f in AUDIO_FEATURES])
    x_audio = audio_scaled * weights

    onehot = np.zeros((1, len(group_columns)), dtype=float)
    group = genre_to_group(song_row.get("track_genre", ""))
    if group in group_columns:
        onehot[0, group_columns.index(group)] = 1.0
    onehot *= group_weight

    vec = np.hstack([x_audio, onehot])
    return pca.transform(vec)[0]
