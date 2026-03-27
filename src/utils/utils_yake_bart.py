import os
import json
from datasets import Dataset
from torch.utils.data import DataLoader


def _load_json_or_jsonl(path):
    """Read either JSON array or JSONL file and return list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        start = f.read(1)
        f.seek(0)
        if start == "[":
            return json.load(f)
        # JSONL fallback
        return [json.loads(line) for line in f if line.strip()]


def get_processed_elife_data(ds_path, tokenizer, config, split, shuffle=False):
    """Load YAKE/BART data directly from the provided path and tokenize."""

    def process_data_to_model_inputs(batch):
        # Validate that articles are not empty
        articles = batch["article"]
        summaries = batch["summary"]
        
        # Check for empty articles
        if isinstance(articles, list):
            empty_indices = [i for i, art in enumerate(articles) if not art or not art.strip()]
            if empty_indices:
                sample_ids = [batch.get("id", ["unknown"])[i] if "id" in batch else "unknown" for i in empty_indices[:3]]
                raise ValueError(
                    f"Found {len(empty_indices)} empty articles in batch. "
                    f"Sample IDs: {sample_ids}. "
                    f"The YAKE data file appears to be corrupted or incomplete. "
                    f"Please re-run the data generation script (create_yake_bart_data.py)."
                )
        elif not articles or not articles.strip():
            raise ValueError(
                f"Empty article found. "
                f"The YAKE data file appears to be corrupted. "
                f"Please re-run create_yake_bart_data.py"
            )
        
        inputs = tokenizer(
            articles,
            padding="max_length",
            truncation=True,
            max_length=config["encoder_max_length"],
        )
        outputs = tokenizer(
            summaries,
            padding="max_length",
            truncation=True,
            max_length=config["decoder_max_length"],
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # global_attention_mask kept for compatibility, but not used in BART
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]
        batch["global_attention_mask"][0][0] = 1

        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in outputs.input_ids
        ]

        return batch

    data = _load_json_or_jsonl(ds_path)

    # Add running index for later bookkeeping/debugging
    data = [{"idx": i, **x} for i, x in enumerate(data)]

    dataset = Dataset.from_list(data)
    dataset = dataset.map(process_data_to_model_inputs, batched=True, batch_size=config["batch_size"])

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "idx"],
    )

    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)

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