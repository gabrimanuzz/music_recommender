"""
main.py
-------
Applicazione grafica principale del Music Recommender.
Usa CustomTkinter: una versione moderna di Tkinter con widget
stilizzati, tema scuro/chiaro e aspetto simile ad app native.

Installa con: pip install customtkinter

Struttura dell'interfaccia:
  ┌─────────────────────────────────────────┐
  │  🎵 MUSIC RECOMMENDER                   │
  ├─────────────────────────────────────────┤
  │  [Barra di ricerca]  [Cerca]            │
  ├──────────────┬──────────────────────────┤
  │ Risultati    │  Raccomandazioni         │
  │ ricerca      │  (dopo selezione)        │
  └──────────────┴──────────────────────────┘
"""

import os
import sys
import customtkinter as ctk
from tkinter import messagebox
import threading  # per non bloccare la GUI durante le operazioni pesanti

# Aggiungiamo la root del progetto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import load_model
from src.recommender import search_songs, get_recommendations


# ─────────────────────────────────────────────
#  Configurazione dell'aspetto globale
# ─────────────────────────────────────────────
# CustomTkinter permette di scegliere il tema ("dark"/"light"/"system")
# e la palette colori ("blue", "green", "dark-blue").
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class MusicRecommenderApp(ctk.CTk):
    """
    Classe principale dell'applicazione.
    Estende ctk.CTk che è la finestra root di CustomTkinter
    (equivalente a tk.Tk() ma con stile moderno).
    """

    def __init__(self):
        super().__init__()  # inizializza la finestra base

        # ── Configurazione finestra ────────────────
        self.title("🎵 Music Recommender — KNN")
        self.geometry("1100x700")
        self.minsize(800, 550)

        # Variabile che terrà la canzone selezionata dall'utente
        # (indice nel DataFrame).
        self.selected_song_index = None

        # ── Caricamento del modello ────────────────
        # Carichiamo il modello all'avvio dell'app.
        # Se il modello non esiste, guidiamo l'utente.
        self._load_model_safe()

        # ── Costruzione interfaccia ────────────────
        self._build_ui()

    # ─────────────────────────────────────────────
    #  CARICAMENTO MODELLO
    # ─────────────────────────────────────────────

    def _load_model_safe(self):
        """Carica il modello con gestione degli errori."""
        try:
            self.model, self.scaler, self.df = load_model("model")
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False

    # ─────────────────────────────────────────────
    #  COSTRUZIONE INTERFACCIA
    # ─────────────────────────────────────────────

    def _build_ui(self):
        """Costruisce tutti i widget dell'interfaccia."""

        # ── Layout principale con grid ─────────────
        # Usiamo grid() invece di pack() per un controllo
        # più preciso sulla disposizione degli elementi.
        # columnconfigure(weight=1) → la colonna si espande
        # con la finestra.
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # la riga 1 si espande

        # ── Header ────────────────────────────────
        self._build_header()

        # ── Corpo principale ──────────────────────
        if not self.model_loaded:
            self._build_error_panel()
        else:
            self._build_main_panel()

    def _build_header(self):
        """Barra in cima con titolo e barra di ricerca."""

        header = ctk.CTkFrame(self, corner_radius=0, height=80)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.grid_columnconfigure(1, weight=1)

        # Titolo
        title = ctk.CTkLabel(
            header,
            text="🎵  Music Recommender",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=20)

        # Barra di ricerca
        # StringVar è una variabile Tkinter che si aggiorna automaticamente
        # quando l'utente scrive nel campo di testo.
        self.search_var = ctk.StringVar()
        self.search_entry = ctk.CTkEntry(
            header,
            textvariable=self.search_var,
            placeholder_text="Cerca un brano o artista...",
            height=38,
            font=ctk.CTkFont(size=14)
        )
        self.search_entry.grid(row=0, column=1, padx=10, pady=20, sticky="ew")

        # Premiamo Invio nella barra di ricerca → stessa azione del bottone
        self.search_entry.bind("<Return>", lambda e: self._on_search())

        # Bottone Cerca
        search_btn = ctk.CTkButton(
            header,
            text="Cerca",
            width=90,
            height=38,
            font=ctk.CTkFont(size=14),
            command=self._on_search  # funzione chiamata al click
        )
        search_btn.grid(row=0, column=2, padx=(0, 20), pady=20)

    def _build_main_panel(self):
        """Pannello centrale con lista risultati e raccomandazioni."""

        # Frame che contiene entrambe le colonne
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=2)  # colonna destra più larga
        main.grid_rowconfigure(1, weight=1)

        # ── Colonna sinistra: risultati ricerca ───
        left_label = ctk.CTkLabel(
            main,
            text="Risultati ricerca",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        left_label.grid(row=0, column=0, sticky="w", padx=5, pady=(0, 5))

        # CTkScrollableFrame: frame con scrollbar automatica.
        # Utile quando i risultati sono tanti.
        self.search_results_frame = ctk.CTkScrollableFrame(main, corner_radius=8)
        self.search_results_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        # ── Colonna destra: raccomandazioni ───────
        right_label = ctk.CTkLabel(
            main,
            text="Brani simili",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        right_label.grid(row=0, column=1, sticky="w", padx=5, pady=(0, 5))

        self.rec_frame = ctk.CTkScrollableFrame(main, corner_radius=8)
        self.rec_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        # Messaggio iniziale nella colonna destra
        self._show_placeholder()

        # ── Status bar in basso ───────────────────
        self.status_var = ctk.StringVar(value=f"Dataset: {len(self.df):,} brani caricati")
        status_bar = ctk.CTkLabel(
            self,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        status_bar.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 8))

    def _build_error_panel(self):
        """Pannello mostrato se il modello non è stato addestrato."""

        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=40, pady=40)

        ctk.CTkLabel(
            frame,
            text="⚠️  Modello non trovato",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(40, 10))

        ctk.CTkLabel(
            frame,
            text="Devi prima addestrare il modello.\n\nPosiziona il dataset in  data/dataset.csv\npoi esegui:",
            font=ctk.CTkFont(size=14)
        ).pack(pady=10)

        ctk.CTkLabel(
            frame,
            text="python train.py",
            font=ctk.CTkFont(size=14, family="Courier"),
            fg_color=("gray85", "gray20"),
            corner_radius=6,
            padx=20, pady=10
        ).pack(pady=5)

    # ─────────────────────────────────────────────
    #  LOGICA DI RICERCA E RACCOMANDAZIONE
    # ─────────────────────────────────────────────

    def _on_search(self):
        """Chiamata quando l'utente preme 'Cerca' o Invio."""

        query = self.search_var.get().strip()
        if not query:
            return

        self.status_var.set("Ricerca in corso...")

        # Usiamo un thread separato per non bloccare la GUI
        # mentre Python scorre il DataFrame (operazione veloce
        # ma su 100k righe può causare un piccolo freeze).
        threading.Thread(target=self._search_worker, args=(query,), daemon=True).start()

    def _search_worker(self, query: str):
        """Worker che gira in un thread separato per la ricerca."""

        results = search_songs(query, self.df, max_results=15)

        # Tkinter non è thread-safe: aggiorniamo la GUI
        # usando after() che schedula l'aggiornamento
        # nel thread principale.
        self.after(0, self._display_search_results, results)

    def _display_search_results(self, results):
        """Mostra i risultati della ricerca nella colonna sinistra."""

        # Svuotiamo il frame precedente prima di inserire i nuovi risultati.
        # winfo_children() restituisce tutti i widget figli del frame.
        for widget in self.search_results_frame.winfo_children():
            widget.destroy()

        if results.empty:
            ctk.CTkLabel(
                self.search_results_frame,
                text="Nessun risultato trovato.",
                text_color="gray"
            ).pack(pady=20)
            self.status_var.set("Nessun risultato.")
            return

        # Creiamo un bottone per ogni canzone trovata.
        # Usiamo lambda con default argument (idx=idx) per
        # "catturare" il valore corrente di idx nel loop.
        # Senza =idx, tutti i bottoni userebbero l'ultimo valore.
        for _, row in results.iterrows():
            idx = row.name  # indice nel DataFrame originale
            self._create_song_button(self.search_results_frame, row, idx, mode="search")

        self.status_var.set(f"Trovati {len(results)} brani.")

    def _create_song_button(self, parent, row, df_index, mode="search"):
        """
        Crea un widget "carta canzone" cliccabile.

        Parameters
        ----------
        parent : widget
            Frame in cui inserire la carta.
        row : pd.Series
            Riga del DataFrame con i dati della canzone.
        df_index : int
            Indice della canzone nel DataFrame.
        mode : str
            "search" = risultato ricerca, "rec" = raccomandazione.
        """

        # Frame esterno che fa da carta
        card = ctk.CTkFrame(parent, corner_radius=8, cursor="hand2")
        card.pack(fill="x", padx=5, pady=3)

        # Titolo della canzone (troncato se troppo lungo)
        name = str(row.get("track_name", "Sconosciuto"))
        if len(name) > 45:
            name = name[:42] + "..."

        artist = str(row.get("artists", ""))
        if len(artist) > 40:
            artist = artist[:37] + "..."

        ctk.CTkLabel(
            card,
            text=name,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        ).pack(fill="x", padx=12, pady=(8, 0))

        ctk.CTkLabel(
            card,
            text=f"🎤 {artist}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 2))

        # Se è una raccomandazione, mostriamo anche la similarità
        if mode == "rec" and "similarity" in row:
            sim = row["similarity"]
            color = "#1DB954" if sim >= 80 else "#FFA500" if sim >= 60 else "gray"
            ctk.CTkLabel(
                card,
                text=f"Similarità: {sim}%",
                font=ctk.CTkFont(size=11),
                text_color=color,
                anchor="w"
            ).pack(fill="x", padx=12, pady=(0, 8))
        else:
            ctk.CTkLabel(
                card,
                text=f"♪ {row.get('track_genre', '')}",
                font=ctk.CTkFont(size=11),
                text_color="#5B8DEF",
                anchor="w"
            ).pack(fill="x", padx=12, pady=(0, 8))

        # Rendiamo cliccabile il frame e tutte le sue label.
        # bind("<Button-1>") intercetta il click sinistro del mouse.
        if mode == "search":
            callback = lambda e, i=df_index: self._on_song_selected(i)
            card.bind("<Button-1>", callback)
            for child in card.winfo_children():
                child.bind("<Button-1>", callback)

    def _on_song_selected(self, df_index: int):
        """Chiamata quando l'utente clicca su una canzone nella lista."""

        self.selected_song_index = df_index
        song = self.df.loc[df_index]
        self.status_var.set(f"Cerco brani simili a: {song['track_name']}...")

        # Calcoliamo le raccomandazioni in un thread separato
        threading.Thread(
            target=self._recommend_worker,
            args=(df_index,),
            daemon=True
        ).start()

    def _recommend_worker(self, df_index: int):
        """Worker per il calcolo delle raccomandazioni KNN."""
        recs = get_recommendations(df_index, self.df, self.model, self.scaler, n_recommendations=10)
        self.after(0, self._display_recommendations, recs, df_index)

    def _display_recommendations(self, recs, original_index: int):
        """Mostra le raccomandazioni nella colonna destra."""

        for widget in self.rec_frame.winfo_children():
            widget.destroy()

        # Mostriamo prima la canzone originale come riferimento
        original = self.df.loc[original_index]
        ctk.CTkLabel(
            self.rec_frame,
            text="Brano selezionato:",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(anchor="w", padx=5, pady=(5, 2))

        orig_card = ctk.CTkFrame(self.rec_frame, corner_radius=8, fg_color=("#1DB954", "#1a7a38"))
        orig_card.pack(fill="x", padx=5, pady=(0, 12))
        ctk.CTkLabel(
            orig_card,
            text=str(original.get("track_name", "")),
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        ).pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(
            orig_card,
            text=f"🎤 {original.get('artists', '')}",
            font=ctk.CTkFont(size=11),
            anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            self.rec_frame,
            text="Brani simili:",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(anchor="w", padx=5, pady=(0, 2))

        for _, row in recs.iterrows():
            self._create_song_button(self.rec_frame, row, row.name, mode="rec")

        self.status_var.set(f"✅ Trovati {len(recs)} brani simili a: {original['track_name']}")

    def _show_placeholder(self):
        """Messaggio iniziale nella colonna raccomandazioni."""
        ctk.CTkLabel(
            self.rec_frame,
            text="🔍\n\nCerca un brano e cliccaci sopra\nper vedere i brani simili",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack(expand=True, pady=80)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = MusicRecommenderApp()
    # mainloop() avvia il loop degli eventi di Tkinter:
    # la finestra rimane aperta e risponde a click, tastiera, ecc.
    # finché l'utente non la chiude.
    app.mainloop()
