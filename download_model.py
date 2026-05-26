import argparse
import os
import sys
import urllib.request

import joblib


DEFAULT_MODEL_URL = "https://github.com/gabrimanuzz/music_recommender/releases/download/1.0/knn_model.pkl"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "model", "knn_model.joblib")


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    # callback di urlretrieve che stampa la percentuale scaricata
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    mb_done = downloaded / (1024 * 1024)
    mb_tot = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  [{pct:3d}%] {mb_done:6.1f} / {mb_tot:.1f} MB")
    sys.stdout.flush()


def download_model(url: str, output_path: str) -> str:
    # scarica il modello e verifica che sia caricabile da joblib
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[1/2] Download da: {url}")
    urllib.request.urlretrieve(url, output_path, reporthook=_progress)
    print()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      Salvato in: {output_path} ({size_mb:.1f} MB)")

    print("[2/2] Verifica con joblib.load()...")
    model = joblib.load(output_path)
    print(f"      OK — oggetto caricato: {type(model).__name__}")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scarica il modello pre-addestrato in formato joblib."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_MODEL_URL,
        help="URL del file .joblib da scaricare.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help="Percorso di destinazione (default: model/knn_model.joblib).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sovrascrivi il file di destinazione se esiste già.",
    )
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"[ERRORE] Il file '{args.out}' esiste già. Usa --force per sovrascriverlo.")
        return 1

    if args.url == "https://example.com/knn_model.joblib":
        print(
            "[ERRORE] URL di default non configurato.\n"
            "        Modifica DEFAULT_MODEL_URL in download_model.py "
            "oppure passa --url <link>."
        )
        return 1

    try:
        download_model(args.url, args.out)
    except Exception as e:
        print(f"\n[ERRORE] Download fallito: {e}")
        # cleanup di eventuale file parziale
        if os.path.exists(args.out):
            os.remove(args.out)
        return 1

    print("\n✅ Modello pronto all'uso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
