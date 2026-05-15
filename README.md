# 🎵 Music Recommender — KNN

Sistema di raccomandazione musicale basato su K-Nearest Neighbors,
addestrato sul dataset Spotify di Kaggle.

## Struttura del progetto

```
music_recommender/
├── data/
│   └── dataset.csv          ← metti qui il CSV di Kaggle
├── model/                   ← creata automaticamente da train.py
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   └── songs_df.pkl
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     ← caricamento, pulizia, normalizzazione
│   ├── model.py             ← build/save/load del modello KNN
│   └── recommender.py       ← logica di ricerca e raccomandazione
├── train.py                 ← script di addestramento (eseguire 1 volta)
├── main.py                  ← applicazione grafica
└── requirements.txt
```

## Setup

### 1. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 2. Scarica il dataset
Vai su: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/
Scarica `dataset.csv` e mettilo nella cartella `data/`.

### 3. Addestra il modello (una sola volta)
```bash
python train.py
```

### 4. Avvia l'applicazione
```bash
python main.py
```

### (Opzionale) Crea un eseguibile .exe
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "MusicRecommender" main.py
```
Il file `.exe` sarà in `dist/MusicRecommender.exe`.

## Come funziona

1. **Preprocessing**: il dataset viene pulito (NaN rimossi, duplicati eliminati) e le 9 feature audio vengono normalizzate con `StandardScaler`.
2. **KNN**: `NearestNeighbors` di scikit-learn memorizza tutte le canzoni nello spazio delle feature. La distanza coseno misura la "direzione" del suono.
3. **Raccomandazione**: selezionata una canzone, il suo vettore viene scalato e passato a `kneighbors()` che restituisce i 10 brani più vicini.

## Feature audio usate

| Feature | Descrizione |
|---|---|
| danceability | Adatta al ballo (0–1) |
| energy | Intensità percepita (0–1) |
| loudness | Volume medio (dB) |
| speechiness | Presenza di parlato (0–1) |
| acousticness | Probabilità acustica (0–1) |
| instrumentalness | Assenza di voci (0–1) |
| liveness | Probabilità live (0–1) |
| valence | Positività emotiva (0–1) |
| tempo | BPM |
