import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu


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


def compute_sentence_bleu(preds, refs):
    scores = []
    for pred, ref in zip(preds, refs):
        score = sacrebleu.sentence_bleu(pred, [ref]).score
        scores.append(score)
    return scores


def build_violin_plot(data_df, output_path, title):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    methods = data_df["method"].unique().tolist()
    palette = {
        "BART": "#1f77b4",
        "GoldSack": "#ff7f0e",
        "YAKE+DBPEDIA": "#2ca02c",
    }

    base_x = 1.0
    offsets = np.linspace(-0.18, 0.18, len(methods))

    for offset, method in zip(offsets, methods):
        values = data_df.loc[data_df["method"] == method, "bleu"].values
        parts = plt.violinplot(
            values,
            positions=[base_x + offset],
            widths=0.3,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(palette.get(method, "#333333"))
            pc.set_edgecolor("black")
            pc.set_alpha(0.45)

        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)

        plt.errorbar(
            base_x + offset,
            mean,
            yerr=std,
            fmt="o",
            color=palette.get(method, "#333333"),
            capsize=6,
            linewidth=1.5,
            label=f"{method} mean ± std",
        )
        plt.scatter(
            base_x + offset,
            median,
            color=palette.get(method, "#333333"),
            marker="D",
            s=50,
            edgecolor="black",
            label=f"{method} median",
        )

    plt.xticks([base_x], ["BLEU"], fontsize=12, weight="bold")
    plt.ylabel("BLEU score (%)", fontsize=12, weight="bold")
    plt.title(title, fontsize=14, weight="bold", pad=12)
    plt.grid(axis="y", alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate legend entries
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base = Path(r"c:\Users\Elisabete\OneDrive\Ambiente de Trabalho\Pedro\Biomedical_Summary_Enhanced")
    output_dir = base / "thesis_visualizations" / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "BART": {
            "path": base / "Resultados Tese" / "BART" / "BestRun" / "predictions.jsonl",
            "loader": load_jsonl_predictions,
        },
        "GoldSack": {
            "path": base / "Resultados Tese" / "GoldSack" / "BestRunTEST" / "test_descriptions_umls.json",
            "loader": load_json_predictions,
        },
        "YAKE+DBPEDIA": {
            "path": base / "Resultados Tese" / "YAKE+DBPEDIA" / "BestRunTEST" / "test_descriptions.json",
            "loader": load_json_predictions,
        },
    }

    records = []
    stats = []

    for method, cfg in sources.items():
        preds, refs = cfg["loader"](cfg["path"])
        scores = compute_sentence_bleu(preds, refs)
        for score in scores:
            records.append({"method": method, "bleu": score})

        stats.append({
            "method": method,
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "count": len(scores),
        })

    df = pd.DataFrame(records)
    stats_df = pd.DataFrame(stats)

    stats_csv = output_dir / "bleu_per_document_stats.csv"
    stats_df.to_csv(stats_csv, index=False, float_format="%.4f")

    plot_path = output_dir / "bleu_overlapping_violin.png"
    build_violin_plot(
        df,
        plot_path,
        "BLEU Distribution per Document (BART vs GoldSack vs YAKE+DBPEDIA)",
    )

    print(f"Saved BLEU violin plot to {plot_path}")
    print(f"Saved BLEU summary stats to {stats_csv}")


if __name__ == "__main__":
    main()
