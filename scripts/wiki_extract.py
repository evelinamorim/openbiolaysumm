import argparse
import json
import random
import re
from datasets import load_dataset

BIO_KEYWORDS = [
    "patient","disease","medical","medicine","clinical","blood","cell","cells","protein","enzyme",
    "genetic","genome","tumor","cancer","neuron","neuronal","brain","virus","bacteria","microbi",
    "vaccine","therapy","symptom","diagnos","patholog","drug","pharmac","RNA","DNA","transcript",
    "immun","epidemi","metast","histolog","oncolog"
]
BIO_RE = re.compile(r"\b(" + "|".join([re.escape(w) for w in BIO_KEYWORDS]) + r")", flags=re.I)

def is_non_biomedical(text: str) -> bool:
    if not text:
        return False
    return BIO_RE.search(text) is None

def compute_target_from_datasets(names):
    total = 0
    for name in names:
        try:
            ds = load_dataset(name, split="train")
            n = len(ds)
            print(f"{name}: {n}")
            total += n
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    print("Computed target size (sum):", total)
    return total

def reservoir_sample_stream(stream_iter, k, max_scan=None, seed=42):
    random.seed(seed)
    reservoir = []
    seen = 0
    for item in stream_iter:
        seen += 1
        if max_scan and seen > max_scan:
            break
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = random.randrange(seen)
            if j < k:
                reservoir[j] = item
    return reservoir, seen

def wiki_stream(dataset_name, config_name, streaming=True):
    if streaming:
        ds = load_dataset(dataset_name, config_name, split="train", streaming=True, trust_remote_code=True)
        for e in ds:
            # some wiki variants use 'text' or 'body' fields; prefer 'text'
            text = e.get("text") or e.get("body") or ""
            title = e.get("title") or ""
            yield {"title": title, "text": text}
    else:
        ds = load_dataset(dataset_name, config_name, split="train", trust_remote_code=True)
        for e in ds:
            text = e.get("text") or e.get("body") or ""
            title = e.get("title") or ""
            yield {"title": title, "text": text}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wikipedia", help="HF dataset id (default wikipedia)")
    p.add_argument("--config", default="20220301.en", help="HF config (default 20220301.en)")
    p.add_argument("--target-size", type=int, default=0, help="Number of non-bio pages to collect")
    p.add_argument("--match-datasets", nargs="+", help="HF dataset ids to sum lengths for target-size (eg: ag_news 20_newsgroups)")
    p.add_argument("--output", default="wiki_nonbio.jsonl", help="Output jsonl file")
    p.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended)")
    p.add_argument("--max-scan", type=int, default=None, help="Max pages to scan (streaming) — leave None to scan until filled")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-download", action="store_true", help="Allow non-streaming download fallback if streaming unsupported")
    args = p.parse_args()

    if args.match_datasets:
        print("Computing target size from:", args.match_datasets)
        args.target_size = compute_target_from_datasets(args.match_datasets)

    if args.target_size <= 0:
        raise SystemExit("Specify --target-size > 0 or use --match-datasets to compute it.")

    print(f"Target non-biomedical pages: {args.target_size}")
    print("Streaming:", args.streaming, "max_scan:", args.max_scan, "seed:", args.seed)

    def filtered_generator():
        try:
            for doc in wiki_stream(args.dataset, args.config, streaming=args.streaming):
                if not doc["text"]:
                    continue
                if is_non_biomedical(doc["title"] or "") and is_non_biomedical(doc["text"]):
                    yield {"title": doc["title"], "text": doc["text"]}
        except Exception as e:
            if not args.allow_download:
                raise
            # fallback: try non-streaming (will attempt download)
            for doc in wiki_stream(args.dataset, args.config, streaming=False):
                if not doc["text"]:
                    continue
                if is_non_biomedical(doc["title"] or "") and is_non_biomedical(doc["text"]):
                    yield {"title": doc["title"], "text": doc["text"]}

    print("Scanning wikipedia and collecting candidates (reservoir sampling)...")
    reservoir, scanned = reservoir_sample_stream(filtered_generator(), args.target_size, max_scan=args.max_scan, seed=args.seed)
    print(f"Scanned {scanned} pages; collected {len(reservoir)} items (requested {args.target_size}).")

    print("Writing output to", args.output)
    with open(args.output, "w", encoding="utf-8") as fh:
        for item in reservoir:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Done.")

if __name__ == "__main__":
    main()