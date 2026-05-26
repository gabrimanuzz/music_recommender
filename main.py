import os
import subprocess
import sys
import customtkinter as ctk
import numpy as np

# matplotlib embedded in tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import load_model
from src.recommender import search_songs, get_recommendations
from src.config import save_weights, get_default_weights
from src.preprocessing import AUDIO_FEATURES
from src.visualization import compute_pca_sample, project_point


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# palette colori per i macrogruppi (matplotlib tab20)
GROUP_COLORS = {
    "pop":          "#1f77b4",
    "rock":         "#d62728",
    "metal":        "#7f0000",
    "electronic":   "#9467bd",
    "hiphop":       "#ff7f0e",
    "rnb_soul":     "#bcbd22",
    "jazz_blues":   "#17becf",
    "folk_country": "#8c564b",
    "classical":    "#e377c2",
    "latin":        "#2ca02c",
    "reggae_carib": "#bd5e00",
    "asian":        "#e7298a",
    "world_other":  "#7f7f7f",
    "mood_func":    "#aec7e8",
    "other":        "#cccccc",
}


class MusicRecommenderApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("🎵 Music Recommender — KNN")
        self.geometry("1200x780")
        self.minsize(900, 600)

        self.selected_song_index = None
        # PCA calcolata lazy al primo accesso al tab
        self._pca_data = None

        self._load_model_safe()
        self._build_ui()

    def _load_model_safe(self):
        try:
            (self.model, self.scaler, self.df,
             self.group_columns, self.weights) = load_model("model")
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False

    def _reload_model(self):
        # invalida la PCA cached dopo un retrain
        self._load_model_safe()
        self._pca_data = None
        if hasattr(self, "status_var"):
            self.status_var.set(f"Modello ricaricato — {len(self.df):,} brani.")

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0, height=60)
        header.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            header, text="🎵  Music Recommender",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left", padx=20, pady=14)

        if not self.model_loaded:
            self._build_error_panel()
            return

        self.tabs = ctk.CTkTabview(self, corner_radius=8)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 0))
        self.tabs.add("Recommender")
        self.tabs.add("Visualizzazione")
        self.tabs.add("Pesi")

        self._build_tab_recommender(self.tabs.tab("Recommender"))
        self._build_tab_visualization(self.tabs.tab("Visualizzazione"))
        self._build_tab_weights(self.tabs.tab("Pesi"))

        # lazy-init della visualization al primo cambio tab
        self.tabs.configure(command=self._on_tab_changed)

        self.status_var = ctk.StringVar(
            value=f"Dataset: {len(self.df):,} brani caricati"
        )
        status_bar = ctk.CTkLabel(
            self, textvariable=self.status_var,
            font=ctk.CTkFont(size=11), text_color="gray",
        )
        status_bar.grid(row=2, column=0, sticky="ew", padx=15, pady=(2, 8))

    def _build_error_panel(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=40, pady=40)
        ctk.CTkLabel(
            frame, text="⚠️  Modello non trovato",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(40, 10))
        ctk.CTkLabel(
            frame,
            text="Devi prima addestrare il modello.\n\n"
                 "Posiziona il dataset in  data/dataset.csv  poi esegui:",
            font=ctk.CTkFont(size=14),
        ).pack(pady=10)
        ctk.CTkLabel(
            frame, text="python train.py",
            font=ctk.CTkFont(size=14, family="Courier"),
            fg_color=("gray85", "gray20"), corner_radius=6,
            padx=20, pady=10,
        ).pack(pady=5)

    def _on_tab_changed(self):
        tab = self.tabs.get()
        if tab == "Visualizzazione" and self._pca_data is None:
            self._compute_pca_and_draw()

    # tab 1: recommender
    def _build_tab_recommender(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        # barra di ricerca
        search_bar = ctk.CTkFrame(parent, fg_color="transparent")
        search_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(8, 10))
        search_bar.grid_columnconfigure(0, weight=1)

        self.search_var = ctk.StringVar()
        self.search_entry = ctk.CTkEntry(
            search_bar, textvariable=self.search_var,
            placeholder_text="Cerca un brano o artista...",
            height=38, font=ctk.CTkFont(size=14),
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.search_entry.bind("<Return>", lambda e: self._on_search())

        ctk.CTkButton(
            search_bar, text="Cerca", width=90, height=38,
            font=ctk.CTkFont(size=14), command=self._on_search,
        ).grid(row=0, column=1)

        # due colonne: risultati ricerca + raccomandazioni
        main = ctk.CTkFrame(parent, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            main, text="Risultati ricerca",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=5, pady=(0, 5))

        self.search_results_frame = ctk.CTkScrollableFrame(main, corner_radius=8)
        self.search_results_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        ctk.CTkLabel(
            main, text="Brani simili",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=1, sticky="w", padx=5, pady=(0, 5))

        self.rec_frame = ctk.CTkScrollableFrame(main, corner_radius=8)
        self.rec_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self._show_placeholder()

    def _on_search(self):
        query = self.search_var.get().strip()
        if not query:
            return
        self.status_var.set("Ricerca in corso...")
        self.update_idletasks()
        try:
            results = search_songs(query, self.df, max_results=15)
            self._display_search_results(results)
        except Exception as e:
            self.status_var.set(f"Errore nella ricerca: {e}")

    def _display_search_results(self, results):
        for w in self.search_results_frame.winfo_children():
            w.destroy()
        if results.empty:
            ctk.CTkLabel(
                self.search_results_frame, text="Nessun risultato trovato.",
                text_color="gray",
            ).pack(pady=20)
            self.status_var.set("Nessun risultato.")
            return
        for _, row in results.iterrows():
            self._create_song_button(self.search_results_frame, row, row.name, mode="search")
        self.status_var.set(f"Trovati {len(results)} brani.")

    def _create_song_button(self, parent, row, df_index, mode="search"):
        card = ctk.CTkFrame(parent, corner_radius=8, cursor="hand2")
        card.pack(fill="x", padx=5, pady=3)

        name = str(row.get("track_name", "Sconosciuto"))
        if len(name) > 45:
            name = name[:42] + "..."
        artist = str(row.get("artists", ""))
        if len(artist) > 40:
            artist = artist[:37] + "..."

        ctk.CTkLabel(
            card, text=name, anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(
            card, text=f"🎤 {artist}", anchor="w",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(fill="x", padx=12, pady=(0, 2))

        if mode == "rec" and "similarity" in row:
            sim = row["similarity"]
            color = "#1DB954" if sim >= 80 else "#FFA500" if sim >= 60 else "gray"
            ctk.CTkLabel(
                card, text=f"Similarità: {sim}%  ·  ♪ {row.get('track_genre','')}",
                font=ctk.CTkFont(size=11), text_color=color, anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 8))
        else:
            ctk.CTkLabel(
                card, text=f"♪ {row.get('track_genre', '')}",
                font=ctk.CTkFont(size=11), text_color="#5B8DEF", anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 8))

        if mode == "search":
            cb = lambda e, i=df_index: self._on_song_selected(i)
            card.bind("<Button-1>", cb)
            for child in card.winfo_children():
                child.bind("<Button-1>", cb)

    def _on_song_selected(self, df_index: int):
        self.selected_song_index = df_index
        song = self.df.loc[df_index]
        self.status_var.set(f"Cerco brani simili a: {song['track_name']}...")
        self.update_idletasks()
        try:
            recs = get_recommendations(
                df_index, self.df, self.model, self.scaler,
                self.group_columns, self.weights, n_recommendations=10,
            )
            self._display_recommendations(recs, df_index)
            # se la PCA è già pronta, aggiorna l'evidenziazione
            if self._pca_data is not None:
                self._draw_pca(highlight_index=df_index, highlight_recs=recs)
        except Exception as e:
            self.status_var.set(f"Errore nel calcolo: {e}")

    def _display_recommendations(self, recs, original_index: int):
        for w in self.rec_frame.winfo_children():
            w.destroy()
        original = self.df.loc[original_index]
        ctk.CTkLabel(
            self.rec_frame, text="Brano selezionato:",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).pack(anchor="w", padx=5, pady=(5, 2))

        orig_card = ctk.CTkFrame(
            self.rec_frame, corner_radius=8, fg_color=("#1DB954", "#1a7a38"),
        )
        orig_card.pack(fill="x", padx=5, pady=(0, 12))
        ctk.CTkLabel(
            orig_card, text=str(original.get("track_name", "")), anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(
            orig_card, text=f"🎤 {original.get('artists', '')}", anchor="w",
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            self.rec_frame, text="Brani simili:",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).pack(anchor="w", padx=5, pady=(0, 2))

        for _, row in recs.iterrows():
            self._create_song_button(self.rec_frame, row, row.name, mode="rec")

        self.status_var.set(
            f"✅ Trovati {len(recs)} brani simili a: {original['track_name']}"
        )

    def _show_placeholder(self):
        ctk.CTkLabel(
            self.rec_frame,
            text="🔍\n\nCerca un brano e cliccaci sopra\nper vedere i brani simili",
            font=ctk.CTkFont(size=14), text_color="gray",
        ).pack(expand=True, pady=80)

    # tab 2: visualizzazione
    def _build_tab_visualization(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(parent, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=(8, 8))

        ctk.CTkLabel(
            toolbar,
            text="Proiezione PCA dello spazio KNN — campione di 3000 brani",
            font=ctk.CTkFont(size=13),
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            toolbar, text="Mostra brano selezionato", width=200,
            command=self._on_show_selected,
        ).pack(side="right", padx=5)

        self.viz_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.viz_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.viz_frame.grid_columnconfigure(0, weight=3)
        self.viz_frame.grid_columnconfigure(1, weight=1)
        self.viz_frame.grid_rowconfigure(0, weight=1)

        # placeholder finché l'utente non apre la tab
        self._viz_placeholder = ctk.CTkLabel(
            self.viz_frame,
            text="Apri questa tab per calcolare la proiezione PCA\n(operazione una tantum, ~2 secondi).",
            font=ctk.CTkFont(size=14), text_color="gray",
        )
        self._viz_placeholder.grid(row=0, column=0, columnspan=2, pady=60)

    def _compute_pca_and_draw(self):
        # calcola PCA al primo accesso e disegna entrambi i grafici
        self.status_var.set("Calcolo PCA in corso...")
        self.update_idletasks()

        sample_df, coords, groups, pca = compute_pca_sample(
            self.df, self.scaler,
            self.weights["feature_weights"],
            self.weights["group_weight"],
            self.group_columns,
            n_sample=3000,
        )
        self._pca_data = {
            "sample_df": sample_df,
            "coords": coords,
            "groups": groups,
            "pca": pca,
        }

        self._viz_placeholder.destroy()

        # figura scatter
        self.scatter_fig = Figure(figsize=(7, 5), dpi=100, facecolor="#1a1a1a")
        self.scatter_ax = self.scatter_fig.add_subplot(111)
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=self.viz_frame)
        self.scatter_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                                 padx=(0, 5))

        # figura bar chart
        self.bar_fig = Figure(figsize=(3.5, 5), dpi=100, facecolor="#1a1a1a")
        self.bar_ax = self.bar_fig.add_subplot(111)
        self.bar_canvas = FigureCanvasTkAgg(self.bar_fig, master=self.viz_frame)
        self.bar_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        self._draw_pca()
        self._draw_distribution()
        self.status_var.set(f"PCA calcolata su {len(sample_df):,} brani.")

    def _draw_pca(self, highlight_index=None, highlight_recs=None):
        if self._pca_data is None:
            return
        ax = self.scatter_ax
        ax.clear()
        ax.set_facecolor("#1a1a1a")

        coords = self._pca_data["coords"]
        groups = self._pca_data["groups"]

        # scatter colorato per macrogruppo
        for g in sorted(set(groups)):
            mask = groups == g
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=GROUP_COLORS.get(g, "#888"),
                       s=8, alpha=0.55, label=g, edgecolors="none")

        # highlight del brano selezionato + vicini
        if highlight_index is not None:
            try:
                xy = project_point(
                    self.df.loc[highlight_index], self.scaler,
                    self.weights["feature_weights"],
                    self.weights["group_weight"],
                    self.group_columns, self._pca_data["pca"],
                )
                ax.scatter([xy[0]], [xy[1]], c="#1DB954", s=180,
                           edgecolors="white", linewidths=2,
                           label="selezionato", zorder=5)
                if highlight_recs is not None and not highlight_recs.empty:
                    rec_coords = [
                        project_point(self.df.loc[i], self.scaler,
                                      self.weights["feature_weights"],
                                      self.weights["group_weight"],
                                      self.group_columns,
                                      self._pca_data["pca"])
                        for i in highlight_recs.index
                    ]
                    rec_coords = np.array(rec_coords)
                    ax.scatter(rec_coords[:, 0], rec_coords[:, 1],
                               facecolors="none", edgecolors="#1DB954",
                               s=80, linewidths=1.5,
                               label="vicini", zorder=4)
            except Exception as e:
                print(f"[WARN] highlight fallito: {e}")

        ax.set_title("Spazio del KNN proiettato in 2D (PCA)",
                     color="white", fontsize=12)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7,
                  facecolor="#222", edgecolor="white", labelcolor="white",
                  ncol=2)
        self.scatter_fig.tight_layout()
        self.scatter_canvas.draw_idle()

    def _draw_distribution(self):
        ax = self.bar_ax
        ax.clear()
        ax.set_facecolor("#1a1a1a")

        counts = self.df["track_genre"].apply(
            lambda g: self._group_of(g)
        ).value_counts().sort_values()

        colors = [GROUP_COLORS.get(g, "#888") for g in counts.index]
        ax.barh(counts.index, counts.values, color=colors)
        ax.set_title("Brani per macrogruppo", color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")
        self.bar_fig.tight_layout()
        self.bar_canvas.draw_idle()

    @staticmethod
    def _group_of(genre):
        from src.preprocessing import genre_to_group
        return genre_to_group(genre)

    def _on_show_selected(self):
        if self._pca_data is None:
            self.status_var.set("Apri prima il tab per calcolare la PCA.")
            return
        if self.selected_song_index is None:
            self.status_var.set("Seleziona prima un brano nella tab Recommender.")
            return
        try:
            recs = get_recommendations(
                self.selected_song_index, self.df, self.model, self.scaler,
                self.group_columns, self.weights, n_recommendations=10,
            )
            self._draw_pca(highlight_index=self.selected_song_index,
                           highlight_recs=recs)
            self.status_var.set("Brano e vicini evidenziati nel grafico.")
        except Exception as e:
            self.status_var.set(f"Errore: {e}")

    # tab 3: pesi
    def _build_tab_weights(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        outer = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
        outer.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            outer, text="Pesi delle feature",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", pady=(4, 2))
        ctk.CTkLabel(
            outer,
            text="Sposta gli slider per cambiare l'importanza relativa di "
                 "ogni feature nel KNN. Poi premi 'Salva e riaddestra'.",
            font=ctk.CTkFont(size=12), text_color="gray", wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(0, 14))

        self.weight_sliders = {}
        self.weight_value_labels = {}

        descriptions = {
            "danceability":     "groove e ballabilità del brano",
            "energy":           "intensità e attività percepita",
            "valence":          "positività emotiva (felice ↔ triste)",
            "acousticness":     "probabilità di suono acustico",
            "instrumentalness": "assenza di voci",
            "tempo":            "velocità in BPM",
            "loudness":         "volume medio (correlato all'energia)",
            "speechiness":      "presenza di parlato",
            "liveness":         "probabilità che sia live",
        }

        row_idx = 2
        for feat in AUDIO_FEATURES:
            self._add_weight_slider(
                outer, row_idx, label=feat, desc=descriptions.get(feat, ""),
                current=self.weights["feature_weights"][feat],
                vmin=0.0, vmax=3.0,
            )
            row_idx += 1

        # separatore + group weight
        sep = ctk.CTkFrame(outer, height=1, fg_color="#444")
        sep.grid(row=row_idx, column=0, sticky="ew", pady=(16, 12))
        row_idx += 1

        ctk.CTkLabel(
            outer, text="Peso del macrogruppo di genere",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=row_idx, column=0, sticky="w")
        row_idx += 1

        ctk.CTkLabel(
            outer,
            text="Quanto il KNN deve 'rispettare' il macrogruppo della "
                 "canzone. Alto → raccomandazioni nello stesso genere; "
                 "basso → più mescolanza fra generi.",
            font=ctk.CTkFont(size=11), text_color="gray",
            wraplength=900, justify="left",
        ).grid(row=row_idx, column=0, sticky="w", pady=(0, 6))
        row_idx += 1

        self._add_weight_slider(
            outer, row_idx, label="__group_weight__",
            desc="forza del peso del macrogruppo (consigliato 2.0 – 3.0)",
            current=self.weights["group_weight"],
            vmin=0.0, vmax=5.0,
            display_label="group_weight",
        )
        row_idx += 1

        btn_row = ctk.CTkFrame(outer, fg_color="transparent")
        btn_row.grid(row=row_idx, column=0, sticky="ew", pady=(20, 8))

        ctk.CTkButton(
            btn_row, text="Salva e riaddestra", width=200, height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1DB954", hover_color="#1a9c47",
            command=self._on_save_and_retrain,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_row, text="Reset ai valori predefiniti",
            width=200, height=40, command=self._on_reset_weights,
        ).pack(side="left", padx=5)

    def _add_weight_slider(self, parent, row_idx, label, desc, current,
                           vmin, vmax, display_label=None):
        # riga con etichetta + slider + valore numerico
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row_idx, column=0, sticky="ew", pady=4)
        frame.grid_columnconfigure(1, weight=1)

        shown = display_label or label
        ctk.CTkLabel(
            frame, text=shown, width=140, anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        slider = ctk.CTkSlider(
            frame, from_=vmin, to=vmax,
            number_of_steps=int((vmax - vmin) * 20),
        )
        slider.set(current)
        slider.grid(row=0, column=1, sticky="ew", padx=8)

        value_label = ctk.CTkLabel(
            frame, text=f"{current:.2f}", width=50,
            font=ctk.CTkFont(size=13, family="Courier"),
        )
        value_label.grid(row=0, column=2, padx=(8, 0))

        def on_change(v, lbl=value_label):
            lbl.configure(text=f"{float(v):.2f}")
        slider.configure(command=on_change)

        if desc:
            ctk.CTkLabel(
                frame, text=desc, anchor="w",
                font=ctk.CTkFont(size=10), text_color="gray",
            ).grid(row=1, column=1, columnspan=2, sticky="w", padx=8)

        self.weight_sliders[label] = slider
        self.weight_value_labels[label] = value_label

    def _read_slider_values(self):
        feature_weights = {
            f: float(self.weight_sliders[f].get()) for f in AUDIO_FEATURES
        }
        group_weight = float(self.weight_sliders["__group_weight__"].get())
        return feature_weights, group_weight

    def _on_reset_weights(self):
        defaults = get_default_weights()
        for f in AUDIO_FEATURES:
            v = defaults["feature_weights"][f]
            self.weight_sliders[f].set(v)
            self.weight_value_labels[f].configure(text=f"{v:.2f}")
        v = defaults["group_weight"]
        self.weight_sliders["__group_weight__"].set(v)
        self.weight_value_labels["__group_weight__"].configure(text=f"{v:.2f}")
        self.status_var.set("Pesi reimpostati ai valori predefiniti (non ancora salvati).")

    def _on_save_and_retrain(self):
        feature_weights, group_weight = self._read_slider_values()
        save_weights(feature_weights, group_weight, model_dir="model")
        self.status_var.set("Pesi salvati. Avvio del retraining...")
        self.update_idletasks()

        # train.py come subprocess per non bloccare la GUI (ed evitare segfault con i thread)
        proc = subprocess.Popen(
            [sys.executable, "train.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )

        self._poll_retrain(proc)

    def _poll_retrain(self, proc):
        # polling del subprocess di training, senza bloccare la GUI
        ret = proc.poll()
        if ret is None:
            # ancora in corso, ricontrolla fra 300ms
            self.after(300, lambda: self._poll_retrain(proc))
            return
        output, _ = proc.communicate()
        if ret == 0:
            self.status_var.set("Retraining completato. Ricarico il modello...")
            self.update_idletasks()
            self._reload_model()
            self.status_var.set(
                "✅ Modello aggiornato con i nuovi pesi. "
                "Riprova una ricerca per vedere l'effetto."
            )
        else:
            self.status_var.set(f"❌ Errore nel training (vedi terminale).")
            print(output)


if __name__ == "__main__":
    app = MusicRecommenderApp()
    app.mainloop()
