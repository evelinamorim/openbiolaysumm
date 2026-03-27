import os
import json
from utils import get_processed_elife_data as _get_processed_elife_data

def get_processed_elife_data(*args, **kwargs):
    return _get_processed_elife_data(*args, **kwargs)

def load_train_config(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def update_config(save_dir, kg_config):
    """
    Safe update: ensure save_dir exists and write kg_config into save_dir/config.json.
    This is a standalone, simple implementation used for the BART-only runs.
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(kg_config, f, indent=2)

def postprocess_text(preds, labels):
    """
    Try to sentence-split with nltk if available (no download). If that fails (offline),
    fallback to a simple regex-based splitter that works without network.
    Returns lists of preds and labels with sentences joined by newlines (ROUGE-friendly).
    """
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]

    try:
        import nltk
        sent_tokenize = nltk.sent_tokenize  # will raise if punkt not installed
        preds = ["\n".join(sent_tokenize(p)) for p in preds]
        labels = ["\n".join(sent_tokenize(l)) for l in labels]
        return preds, labels
    except Exception:
        import re
        splitter = lambda s: [seg.strip() for seg in re.split(r'(?<=[\.\?\!])\s+', s) if seg.strip()]
        preds = ["\n".join(splitter(p)) for p in preds]
        labels = ["\n".join(splitter(l)) for l in labels]
        return preds, labels