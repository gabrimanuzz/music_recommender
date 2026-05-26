import json
import os
from src.preprocessing import FEATURE_WEIGHTS, GROUP_WEIGHT, AUDIO_FEATURES


WEIGHTS_FILENAME = "weights.json"


def get_default_weights() -> dict:
    # copia dei pesi default (modificabile senza side-effects)
    return {
        "feature_weights": {k: float(v) for k, v in FEATURE_WEIGHTS.items()},
        "group_weight":    float(GROUP_WEIGHT),
    }


def load_weights(model_dir: str = "model") -> dict:
    # carica i pesi dal file, fallback ai default se manca o è invalido
    path = os.path.join(model_dir, WEIGHTS_FILENAME)
    if not os.path.exists(path):
        return get_default_weights()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # sanity check: tutte le feature audio devono esserci
        fw = data.get("feature_weights", {})
        for feat in AUDIO_FEATURES:
            if feat not in fw:
                return get_default_weights()
        if "group_weight" not in data:
            return get_default_weights()
        return data
    except (json.JSONDecodeError, IOError):
        return get_default_weights()


def save_weights(feature_weights: dict, group_weight: float,
                 model_dir: str = "model") -> None:
    # scrive i pesi su disco, crea la cartella se serve
    os.makedirs(model_dir, exist_ok=True)
    data = {
        "feature_weights": {k: float(v) for k, v in feature_weights.items()},
        "group_weight":    float(group_weight),
    }
    path = os.path.join(model_dir, WEIGHTS_FILENAME)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
