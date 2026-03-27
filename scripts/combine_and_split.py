import os
import json
import glob
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from nltk import word_tokenize

def read_json_or_jsonl(path: str) -> List[dict]:
    rows = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        first = fh.read(1024)
        fh.seek(0)
        if first.lstrip().startswith("["):
            # JSON array
            arr = json.load(fh)
            for j in arr:
                rows.append(j)
        else:
            # JSONL or single-JSON-per-file
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows

def read_elife_source(path_or_dir: str) -> List[dict]:
    rows = []
    p = Path(path_or_dir)
    if p.is_dir():
        files = sorted(p.glob("*.json*"))
    else:
        files = [p]
    idx = 0
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            first = fh.read(1024)
            fh.seek(0)
            if first.lstrip().startswith("["):
                arr = json.load(fh)
                for j in arr:
                    rows.append(normalize_elife(j, idx)); idx += 1
            else:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    j = json.loads(line)
                    rows.append(normalize_elife(j, idx)); idx += 1
    return rows

def normalize_elife(j: dict, idx: int) -> dict:
    text = " ".join(filter(None, [
        j.get("title") if isinstance(j.get("title"), str) else "",
        " ".join(j.get("abstract")) if isinstance(j.get("abstract"), list) else "",
        j.get("body") if isinstance(j.get("body"), str) else "",
        j.get("text") if isinstance(j.get("text"), str) else "",
    ])).strip()
    if not text:
        # fallback: try entire JSON as string
        text = json.dumps(j, ensure_ascii=False)
    return {"id": f"elife_{idx}", "text": text, "label": 1, "source": "elife"}

def normalize_other(j: dict, idx: int, prefix: str, source: str) -> dict:
    text = ""
    if isinstance(j.get("text"), str):
        text = j.get("text")
    else:
        text = " ".join(filter(None, [j.get("title","") if isinstance(j.get("title"), str) else "",
                                      j.get("description","") if isinstance(j.get("description"), str) else ""]))
    text = text.strip() or json.dumps(j, ensure_ascii=False)
    return {"id": f"{prefix}_{idx}", "text": text, "label": 0, "source": source}

def main(args):
    os.makedirs(args.out, exist_ok=True)

    print("Reading eLife from", args.elife)
    elife_rows = read_elife_source(args.elife)
    print("eLife records:", len(elife_rows))
    print("Reading AG News from", args.ag)
    ag_rows_raw = read_json_or_jsonl(args.ag)
    ag_rows = [normalize_other(j, i, "ag", "ag_news") for i, j in enumerate(ag_rows_raw)]
    print("AG records:", len(ag_rows))

    print("Reading 20NG from", args.ng)
    ng_rows_raw = read_json_or_jsonl(args.ng)
    ng_rows = [normalize_other(j, i, "20ng", "20newsgroups") for i, j in enumerate(ng_rows_raw)]
    print("20NG records:", len(ng_rows))

    all_rows = elife_rows + ag_rows + ng_rows
    df = pd.DataFrame(all_rows)[["id","text","label","source"]]
    print("Combined total:", len(df))
    print("Class counts before balancing:\n", df.label.value_counts())

    # optional balancing (downsample non-bio to match bio)
    if args.balance == "downsample":
        n_bio = df.label.value_counts().get(1, 0)
        df_bio = df[df.label==1]
        df_non = df[df.label==0].sample(n=min(n_bio, len(df[df.label==0])), random_state=args.seed)
        df = pd.concat([df_bio, df_non]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print("After downsample class counts:\n", df.label.value_counts())
    elif args.balance == "none":
        pass
    else:
        raise ValueError("balance must be 'none' or 'downsample'")

    # stratified split
    train, test = train_test_split(df, test_size=args.test_size, stratify=df["label"], random_state=args.seed)
    print("Train/test sizes:", len(train), len(test))
    # save JSONL and CSV
    def save_df(df_obj, base_name):
        jsonl_path = os.path.join(args.out, base_name + ".jsonl")
        csv_path = os.path.join(args.out, base_name + ".csv")
        df_obj.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        df_obj.to_csv(csv_path, index=False)
        print("Saved", jsonl_path, csv_path)

    save_df(pd.DataFrame(all_rows), "combined")
    save_df(train, "train")
    save_df(test, "test")
    meta = {
        "total": len(df),
        "train": len(train),
        "test": len(test),
        "class_counts": df.label.value_counts().to_dict()
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("Wrote outputs to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--elife", required=True, help="Path to eLife file or directory of eLife JSON/JSONL files")
    p.add_argument("--ag", required=True, help="Path to ag_news JSON or JSONL (from your scripts folder)")
    p.add_argument("--ng", required=True, help="Path to 20newsgroups JSON or JSONL")
    p.add_argument("--out", default="./data", help="Output folder for combined/train/test")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance", choices=["none","downsample"], default="none",
                   help="If 'downsample', reduce non-bio samples to match bio count")
    args = p.parse_args()
    main(args)
