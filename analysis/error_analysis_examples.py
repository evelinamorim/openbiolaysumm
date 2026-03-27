import json
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import textstat
import matplotlib.pyplot as plt


def load_json_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("predictions", [])
    refs = data.get("references", [])
    if not preds or not refs:
        raise ValueError(f"No predictions/references found in {path}")
    if len(preds) != len(refs):
        raise ValueError(f"Predictions and references length mismatch in {path}")
    return preds, refs


def load_jsonl_predictions(path):
    preds = []
    refs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            preds.append(item.get("pred", ""))
            refs.append(item.get("ref", ""))
    if not preds or not refs:
        raise ValueError(f"No predictions/references found in {path}")
    if len(preds) != len(refs):
        raise ValueError(f"Predictions and references length mismatch in {path}")
    return preds, refs


def load_test_json(path):
    if not path or not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def looks_like_test_split(data):
    if not isinstance(data, list) or not data:
        return False
    sample = data[0]
    return isinstance(sample, dict) and "abstract" in sample and "summary" in sample


def find_test_json(base_path):
    # Common candidate locations
    candidates = [
        base_path / "test.json",
        base_path / "DSplit" / "test.json",
        base_path / "DSplit" / "elife_test.json",
        base_path / "DSplit" / "elife_umls_test.json",
    ]

    for path in candidates:
        if path.exists():
            data = load_test_json(path)
            if looks_like_test_split(data):
                print(f"Using test split: {path}")
                return data

    # Fallback: search for any *test*.json containing abstracts
    for path in base_path.rglob("*test*.json"):
        try:
            data = load_test_json(path)
            if looks_like_test_split(data):
                print(f"Using test split: {path}")
                return data
        except Exception:
            continue

    print("Warning: No test split JSON with abstracts found.")
    return None


def compute_sentence_bleu(preds, refs):
    scores = []
    for pred, ref in zip(preds, refs):
        score = sacrebleu.sentence_bleu(pred, [ref]).score
        scores.append(score)
    return np.array(scores)


def sentence_split(text):
    # Simple sentence split to avoid extra deps
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return sentences if sentences else [text]


def word_split(text):
    return [w for w in text.replace("\n", " ").split() if w.strip()]


def readability_metrics(text):
    sentences = sentence_split(text)
    words = word_split(text)
    if not words:
        return {
            "flesch": 0.0,
            "coleman_liau": 0.0,
            "dale_chall": 0.0,
            "avg_sentence_len": 0.0,
            "avg_word_len": 0.0,
            "ttr": 0.0,
        }

    flesch = textstat.flesch_reading_ease(text)
    coleman_liau = textstat.coleman_liau_index(text)
    dale_chall = textstat.dale_chall_readability_score(text)
    avg_sentence_len = len(words) / max(1, len(sentences))
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    ttr = len(set(w.lower() for w in words)) / max(1, len(words))

    return {
        "flesch": flesch,
        "coleman_liau": coleman_liau,
        "dale_chall": dale_chall,
        "avg_sentence_len": avg_sentence_len,
        "avg_word_len": avg_word_len,
        "ttr": ttr,
    }


def aggregate_readability(texts):
    rows = [readability_metrics(t) for t in texts]
    df = pd.DataFrame(rows)
    return df.mean().to_dict(), df.std().to_dict(), df


def create_readability_boxplots(readability_by_method, output_path):
    metrics = [
        ("flesch", "Flesch Reading Ease"),
        ("coleman_liau", "Coleman-Liau Index"),
        ("dale_chall", "Dale-Chall Score"),
        ("avg_sentence_len", "Avg Sentence Length"),
        ("avg_word_len", "Avg Word Length"),
        ("ttr", "Type-Token Ratio"),
    ]

    method_names = list(readability_by_method.keys())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        data = [
            readability_by_method[method][metric_key].values
            for method in method_names
        ]
        ax.boxplot(data, labels=method_names, patch_artist=True)
        ax.set_title(metric_title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def normalize_text(text):
    return " ".join(text.lower().split())


def build_summary_lookup(meta):
    if not meta:
        return {}
    lookup = {}
    for item in meta:
        summary_list = item.get("summary", [])
        summary_text = " ".join(summary_list)
        lookup[normalize_text(summary_text)] = item
    return lookup


def load_additional_splits(base_path):
    extra = []
    for fname in ["val.json", "train_filtered.json", "train.json"]:
        path = base_path / fname
        if path.exists():
            data = load_test_json(path)
            if looks_like_test_split(data):
                print(f"Loaded additional split: {path}")
                extra.extend(data)
    return extra


def load_keyword_files(paths):
    keywords_by_id = {}
    for path in paths:
        if not path or not path.exists():
            continue
        data = load_test_json(path)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not item_id:
                continue
            kws = item.get("keywords", [])
            if not isinstance(kws, list):
                continue
            existing = keywords_by_id.get(item_id, [])
            merged = existing + [kw for kw in kws if kw not in existing]
            keywords_by_id[item_id] = merged
    return keywords_by_id


def classify_subarea(text, keywords):
    combined = " ".join([text or "", " ".join(keywords or [])]).lower()
    rules = [
        ("Neuroscience", [
            "neuron", "brain", "cortex", "synapse", "synaptic", "neural",
            "glia", "hippocamp", "cerebell", "olfactory", "visual", "auditory",
        ]),
        ("Immunology", [
            "immune", "immun", "t cell", "b cell", "antibody", "cytokine",
            "inflamm", "vaccin", "antigen", "pathogen",
        ]),
        ("Genetics/Genomics", [
            "genome", "genetic", "dna", "rna", "chromosome", "mutation",
            "transcription", "sequenc", "epigen", "methyl",
        ]),
        ("Cancer/Oncology", [
            "cancer", "tumor", "tumour", "oncogen", "metast", "carcin",
        ]),
        ("Microbiology/Virology", [
            "bacter", "microb", "virus", "viral", "fung", "parasite",
            "pathogen", "tb", "tuberculosis", "yeast",
        ]),
        ("Developmental Biology", [
            "development", "embryo", "embryonic", "differenti", "stem cell",
            "morphogen", "somit", "myogenesis",
        ]),
        ("Physiology/Systems", [
            "physiology", "metabolism", "homeostasis", "endocr", "hormone",
            "cardio", "respirat", "renal", "liver", "muscle",
        ]),
        ("Clinical/Translational", [
            "patient", "clinical", "trial", "cohort", "diagnos", "therapy",
            "treatment", "disease", "disorder",
        ]),
    ]

    for label, terms in rules:
        if any(term in combined for term in terms):
            return label
    return "Unknown"


def pick_examples(
    gold_bleu,
    yake_bleu,
    refs,
    gold_preds,
    yake_preds,
    meta=None,
    keywords_by_id=None,
):
    diff = gold_bleu - yake_bleu
    mean_gold = gold_bleu.mean()
    std_gold = gold_bleu.std()
    mean_yake = yake_bleu.mean()
    std_yake = yake_bleu.std()

    # 1) GoldSack > YAKE (top 3 by diff)
    idx_gold_better = np.argsort(diff)[-3:][::-1]

    # 2) GoldSack < YAKE (top 3 by diff negative)
    idx_yake_better = np.argsort(diff)[:3]

    # 3) Similar (abs diff <= 1 and close to mean)
    similar_mask = (np.abs(diff) <= 1.0) & \
        (np.abs(gold_bleu - mean_gold) <= std_gold) & \
        (np.abs(yake_bleu - mean_yake) <= std_yake)
    idx_similar = np.where(similar_mask)[0][:3]

    # 4) Both low (bottom 10th percentile for both)
    low_gold = np.percentile(gold_bleu, 10)
    low_yake = np.percentile(yake_bleu, 10)
    idx_low_both = np.where((gold_bleu <= low_gold) & (yake_bleu <= low_yake))[0][:3]

    summary_lookup = build_summary_lookup(meta)
    extra_summaries = load_additional_splits(Path(r"c:\Users\Elisabete\OneDrive\Ambiente de Trabalho\Pedro\Biomedical_Summary_Enhanced"))
    if extra_summaries:
        summary_lookup.update(build_summary_lookup(extra_summaries))

    def build_rows(indices, label):
        rows = []
        for i in indices:
            row = {
                "group": label,
                "index": int(i),
                "bleu_goldsack": float(gold_bleu[i]),
                "bleu_yake": float(yake_bleu[i]),
                "reference": refs[i],
                "goldsack_output": gold_preds[i],
                "yake_output": yake_preds[i],
            }
            if summary_lookup:
                key = normalize_text(refs[i])
                item = summary_lookup.get(key)
                if not item and meta is not None and i < len(meta):
                    item = meta[i]
                if item:
                    row["id"] = item.get("id")
                    row["title"] = item.get("title")
                    row["abstract"] = " ".join(item.get("abstract", []))
                    if keywords_by_id and row["id"] in keywords_by_id:
                        row["keywords"] = keywords_by_id[row["id"]]
                    row["subarea"] = classify_subarea(
                        " ".join([row.get("title") or "", row.get("abstract") or ""]),
                        row.get("keywords"),
                    )
            rows.append(row)
        return rows

    rows = []
    rows += build_rows(idx_gold_better, "GoldSack > YAKE")
    rows += build_rows(idx_yake_better, "GoldSack < YAKE")
    rows += build_rows(idx_similar, "GoldSack ≈ YAKE")
    rows += build_rows(idx_low_both, "Both low")

    return rows


def main():
    base = Path(r"c:\Users\Elisabete\OneDrive\Ambiente de Trabalho\Pedro\Biomedical_Summary_Enhanced")
    output_dir = base / "thesis_visualizations" / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    keyword_paths = [
        base / "YakePreProcess" / "File_pre_processed" / "processed_test_keywords.json",
        base / "YakePreProcess" / "File_pre_processed" / "processed_val_keywords.json",
        base / "YakePreProcess" / "Files_pre_processed" / "processed_train_keywords.json",
    ]
    keywords_by_id = load_keyword_files(keyword_paths)

    # Locate test split (must contain abstract + summary)
    test_data = find_test_json(base)

    # GoldSack & YAKE predictions
    gold_path = base / "Resultados Tese" / "GoldSack" / "BestRunTEST" / "test_descriptions_umls.json"
    yake_path = base / "Resultados Tese" / "YAKE+DBPEDIA" / "BestRunTEST" / "test_descriptions.json"

    gold_preds, gold_refs = load_json_predictions(gold_path)
    yake_preds, yake_refs = load_json_predictions(yake_path)

    # Align lengths in case of mismatch
    min_len = min(len(gold_preds), len(gold_refs), len(yake_preds), len(yake_refs))
    if len(gold_refs) != len(yake_refs) or len(gold_preds) != len(yake_preds):
        print(
            "Warning: GoldSack and YAKE lengths do not match. "
            f"Truncating to min length = {min_len}."
        )

    gold_preds = gold_preds[:min_len]
    gold_refs = gold_refs[:min_len]
    yake_preds = yake_preds[:min_len]
    yake_refs = yake_refs[:min_len]

    # BLEU per document
    # Use the same reference set for fair comparison
    gold_bleu = compute_sentence_bleu(gold_preds, gold_refs)
    yake_bleu = compute_sentence_bleu(yake_preds, gold_refs)

    examples = pick_examples(
        gold_bleu,
        yake_bleu,
        gold_refs,
        gold_preds,
        yake_preds,
        meta=test_data,
        keywords_by_id=keywords_by_id,
    )

    examples_path = output_dir / "bleu_examples.json"
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    # Readability stats (Reference, BART, GoldSack, YAKE)
    bart_path = base / "Resultados Tese" / "BART" / "BestRun" / "predictions.jsonl"
    bart_preds, bart_refs = load_jsonl_predictions(bart_path)

    # Use the references from BART (same order)
    ref_texts = bart_refs
    bart_texts = bart_preds
    gold_texts = gold_preds
    yake_texts = yake_preds

    read_summary = []
    readability_by_method = {}
    for name, texts in [
        ("Reference", ref_texts),
        ("BART", bart_texts),
        ("GoldSack", gold_texts),
        ("YAKE+DBPEDIA", yake_texts),
    ]:
        mean_vals, std_vals, read_df = aggregate_readability(texts)
        readability_by_method[name] = read_df
        read_summary.append({
            "method": name,
            "flesch_mean": mean_vals["flesch"],
            "flesch_std": std_vals["flesch"],
            "coleman_liau_mean": mean_vals["coleman_liau"],
            "coleman_liau_std": std_vals["coleman_liau"],
            "dale_chall_mean": mean_vals["dale_chall"],
            "dale_chall_std": std_vals["dale_chall"],
            "avg_sentence_len_mean": mean_vals["avg_sentence_len"],
            "avg_sentence_len_std": std_vals["avg_sentence_len"],
            "avg_word_len_mean": mean_vals["avg_word_len"],
            "avg_word_len_std": std_vals["avg_word_len"],
            "ttr_mean": mean_vals["ttr"],
            "ttr_std": std_vals["ttr"],
        })

    read_df = pd.DataFrame(read_summary)
    read_path = output_dir / "readability_summary.csv"
    read_df.to_csv(read_path, index=False, float_format="%.4f")

    boxplot_path = output_dir / "readability_boxplots.png"
    create_readability_boxplots(readability_by_method, boxplot_path)

    print(f"Saved BLEU example cases to {examples_path}")
    print(f"Saved readability summary to {read_path}")
    print(f"Saved readability boxplots to {boxplot_path}")


if __name__ == "__main__":
    main()
