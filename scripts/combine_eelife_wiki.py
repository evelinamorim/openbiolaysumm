import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import csv

def load_json_or_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    suf = path.suffix.lower()
    if suf == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, list) else [data]
    # assume jsonl
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

def normalize_elife_item(j: Dict) -> Dict:
    text = " ".join(filter(None, [
        j.get("title") if isinstance(j.get("title"), str) else "",
        j.get("abstract") if isinstance(j.get("abstract"), str) else "",
        j.get("body") if isinstance(j.get("body"), str) else "",
        j.get("text") if isinstance(j.get("text"), str) else "",
    ])).strip()
    if not text:
        text = json.dumps(j, ensure_ascii=False)
    summary = j.get("summary") or j.get("abstract") or j.get("summary_text") or ""
    out = {"id": j.get("id", ""), "text": text, "summary": summary, "label": 1, "source": "elife"}
    if "title" in j:
        out["title"] = j.get("title")
    return out

def normalize_wiki_item(j: Dict, idx:int) -> Dict:
    title = j.get("title","")
    text = j.get("text","")
    combined_text = (title + "\n\n" + text).strip() if title else text
    return {"id": f"wiki_{idx}", "text": combined_text, "summary": "", "label": 0, "source": "wikipedia", "title": title}

def write_csv(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=["id","text","summary","label","source","title"])
        writer.writeheader()
        for r in items:
            writer.writerow({
                "id": r.get("id",""),
                "text": r.get("text",""),
                "summary": r.get("summary",""),
                "label": r.get("label",""),
                "source": r.get("source",""),
                "title": r.get("title",""),
            })

def main():
    p = argparse.ArgumentParser(description="Combine eLife splits with wiki non-bio pages and produce combined splits.")
    p.add_argument("--elife-dir", required=True, help="Directory with train.json / val.json / test.json (or .jsonl)")
    p.add_argument("--wiki", required=True, help="Wiki non-bio jsonl file (one JSON per line)")
    p.add_argument("--out-dir", required=True, help="Output directory for combined splits")
    p.add_argument("--target-nonbio-total", type=int, default=0, help="Total wiki items to use (0 = use all available)")
    p.add_argument("--balance", choices=["none","downsample"], default="none", help="If 'downsample', reduce non-bio to match bio count in combined dataset")
    p.add_argument("--shuffle", action="store_true", help="Shuffle each combined split or final combined file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single-split", action="store_true", help="Combine everything into one file then split into train/test by --train-frac")
    p.add_argument("--train-frac", type=float, default=0.8, help="Fraction for train when --single-split is used")
    args = p.parse_args()

    random.seed(args.seed)
    elife_dir = Path(args.elife_dir)
    wiki_path = Path(args.wiki)
    out_dir = Path(args.out_dir)

    # load elife splits (if present)
    splits = {}
    for name in ("train","val","test"):
        for ext in (".json", ".jsonl"):
            pth = elife_dir / f"{name}{ext}"
            if pth.exists():
                splits[name] = load_json_or_jsonl(pth)
                break
        else:
            splits[name] = []

    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    n_elife = n_train + n_val + n_test
    if n_elife == 0:
        raise SystemExit("No elife examples found in elife-dir")

    print(f"Loaded eLife splits: train={n_train}, val={n_val}, test={n_test} (total={n_elife})")

    # load wiki items
    raw_wiki = load_json_or_jsonl(wiki_path)
    n_wiki = len(raw_wiki)
    print(f"Loaded wiki items: {n_wiki}")

    target_total = args.target_nonbio_total if args.target_nonbio_total > 0 else n_wiki
    print(f"Target wiki items to use: {target_total}")

    # sample wiki items without replacement where possible
    if target_total <= n_wiki:
        wiki_sample = random.sample(raw_wiki, target_total)
    else:
        wiki_sample = raw_wiki.copy()
        need = target_total - n_wiki
        wiki_sample += [random.choice(raw_wiki) for _ in range(need)]

    # normalize all elife items and wiki items into a single list
    elife_all = [normalize_elife_item(j) for part in ("train","val","test") for j in splits.get(part,[])]
    wiki_norm = [normalize_wiki_item(w, i) for i, w in enumerate(wiki_sample)]

    full_combined = elife_all + wiki_norm
    print(f"Initial combined size (elife + wiki): {len(full_combined)} (elife {len(elife_all)}, wiki {len(wiki_norm)})")

    # optional downsample non-bio to match bio count (applies to full combined)
    if args.balance == "downsample":
        bio = [r for r in full_combined if r.get("label")==1]
        nonbio = [r for r in full_combined if r.get("label")==0]
        n_bio = len(bio)
        print(f"Before downsample: bio={n_bio}, nonbio={len(nonbio)}")
        if n_bio == 0:
            print("Warning: no bio examples found, skipping downsample")
        else:
            if len(nonbio) > n_bio:
                nonbio = random.sample(nonbio, n_bio)
            full_combined = bio + nonbio
            print(f"After downsample: total={len(full_combined)} (bio={len(bio)}, nonbio={len(nonbio)})")

    # shuffle full combined if requested
    if args.shuffle:
        random.shuffle(full_combined)

    out_dir.mkdir(parents=True, exist_ok=True)

    # If single-split requested -> split into train/test by train-frac (ignore original elife splits)
    if args.single_split:
        total = len(full_combined)
        train_n = int(round(total * args.train_frac))
        train_items = full_combined[:train_n]
        test_items = full_combined[train_n:]
        print(f"Single-split mode: train={len(train_items)}, test={len(test_items)} (train_frac={args.train_frac})")

        write_jsonl(out_dir / "train.jsonl", train_items)
        write_csv(out_dir / "train.csv", train_items)
        write_jsonl(out_dir / "test.jsonl", test_items)
        write_csv(out_dir / "test.csv", test_items)
        write_jsonl(out_dir / "combined.jsonl", full_combined)
        write_csv(out_dir / "combined.csv", full_combined)

        meta = {
            "mode": "single_split",
            "train_frac": args.train_frac,
            "total": total,
            "train": len(train_items),
            "test": len(test_items),
            "elife": {"train": n_train, "val": n_val, "test": n_test, "total": n_elife},
            "wiki_available": n_wiki,
            "wiki_used": target_total,
            "balance": args.balance,
            "shuffle": args.shuffle,
            "seed": args.seed
        }
        with (out_dir / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        print("Wrote single-split outputs to", out_dir)
        return

    # Otherwise preserve previous behaviour: allocate wiki proportionally to original elife splits
    proportions = {"train": n_train / n_elife, "val": n_val / n_elife, "test": n_test / n_elife}
    alloc = {}
    remaining = target_total
    for i, name in enumerate(("train","val","test")):
        if i < 2:
            count = int(round(target_total * proportions[name]))
            alloc[name] = count
            remaining -= count
        else:
            alloc[name] = max(0, remaining)

    print("Wiki allocation per split:", alloc)

    # build combined splits (proportional allocation)
    wiki_idx = 0
    combined_splits = {}
    for name in ("train","val","test"):
        k = alloc.get(name, 0)
        sampled = []
        for _ in range(k):
            if wiki_idx < len(wiki_sample):
                sampled.append(wiki_sample[wiki_idx])
                wiki_idx += 1
            else:
                sampled.append(random.choice(wiki_sample))
        normalized_wiki = [normalize_wiki_item(w, i) for i, w in enumerate(sampled)]
        normalized_elife = [normalize_elife_item(j) for j in splits.get(name,[])]
        combined = normalized_elife + normalized_wiki
        if args.shuffle:
            random.shuffle(combined)
        combined_splits[name] = combined
        print(f"Split {name}: elife {len(normalized_elife)} + wiki {len(normalized_wiki)} -> combined {len(combined)}")

    # produce full combined and write per-split outputs
    full_combined = combined_splits["train"] + combined_splits["val"] + combined_splits["test"]

    for name in ("train","val","test"):
        write_jsonl(out_dir / f"{name}.jsonl", combined_splits[name])
        write_csv(out_dir / f"{name}.csv", combined_splits[name])

    write_jsonl(out_dir / "combined.jsonl", full_combined)
    write_csv(out_dir / "combined.csv", full_combined)

    meta = {
        "mode": "proportional_splits",
        "elife": {"train": n_train, "val": n_val, "test": n_test, "total": n_elife},
        "wiki_available": n_wiki,
        "wiki_used": target_total,
        "combined": {"train": len(combined_splits["train"]), "val": len(combined_splits["val"]), "test": len(combined_splits["test"]), "total": len(full_combined)},
        "balance": args.balance,
        "shuffle": args.shuffle,
        "seed": args.seed
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("Wrote combined splits and meta to", out_dir)

if __name__ == "__main__":
    main()