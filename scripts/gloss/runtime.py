from config.config import (
    custom_gloss_models_path,
    custom_index_map,
    gloss_models_path,
    index_map,
)
from scripts.gloss.backbone import TFLiteModel, get_model
from scripts.gloss.landmarks_extraction import load_json_file


GLOSS_PROFILES = {
    "original": {
        "display_name": "Original pre-trained",
        "index_map_path": index_map,
        "model_paths": gloss_models_path,
    },
    "custom": {
        "display_name": "Custom 25-sign",
        "index_map_path": custom_index_map,
        "model_paths": custom_gloss_models_path,
    },
}


def get_gloss_profile_names():
    return list(GLOSS_PROFILES.keys())


def load_gloss_runtime(profile_name="original"):
    if profile_name not in GLOSS_PROFILES:
        available = ", ".join(get_gloss_profile_names())
        raise ValueError(f"Unknown gloss profile '{profile_name}'. Available: {available}")

    settings = GLOSS_PROFILES[profile_name]
    sign_map = load_json_file(settings["index_map_path"])
    s2p_map = {key.lower(): value for key, value in sign_map.items()}
    p2s_map = {value: key for key, value in sign_map.items()}

    models = [get_model(max_len=None, num_classes=len(s2p_map)) for _ in settings["model_paths"]]
    for model, path in zip(models, settings["model_paths"]):
        model.load_weights(path)

    return {
        "profile_name": profile_name,
        "display_name": settings["display_name"],
        "index_map_path": settings["index_map_path"],
        "model_paths": list(settings["model_paths"]),
        "s2p_map": s2p_map,
        "p2s_map": p2s_map,
        "tflite_model": TFLiteModel(islr_models=models),
    }