import os
import json
import argparse
import re
from typing import Any, Dict, List, Optional

def load_json(path):
    if not os.path.exists(path):
        print(f"Warning: file not found, using empty list: {path}")
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def normalize_kw(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def build_bigrams_index(bigrams: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    """
    Returns mapping: index[article_id][normalized_keyword] -> {"description":..., "link":...}
    """
    index: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    for entry in bigrams:
        aid = entry.get("id") or entry.get("article_id") or entry.get("doc_id")
        if not aid:
            continue
        index.setdefault(aid, {})
        # common layout: entry['dbpedia'] -> { phrase: { "description": "...", ... } }
        db = entry.get("dbpedia") or {}
        for phrase, pdata in db.items():
            if not isinstance(phrase, str):
                continue
            nk = normalize_kw(phrase)
            desc = None
            link = None
            if isinstance(pdata, dict):
                # direct fields
                desc = pdata.get("description") or pdata.get("abstract") or pdata.get("desc")
                link = pdata.get("link") or pdata.get("url") or pdata.get("href")
                # also try to find http-like strings anywhere inside pdata values
                if not link:
                    for v in pdata.values():
                        if isinstance(v, str) and re.search(r"https?://", v):
                            link = re.search(r"https?://\S+", v).group(0)
                            break
                # if no description found but there are string fields, choose the longest
                if not desc:
                    candidates = [v for v in pdata.values() if isinstance(v, str) and len(v) > 20]
                    if candidates:
                        desc = max(candidates, key=len)
            else:
                # pdata not dict: try use it as description if it's a string
                if isinstance(pdata, str):
                    desc = pdata
            index[aid].setdefault(nk, {"description": desc or "", "link": link or ""})
    return index

def merge_keywords_file(key_file: str, bigrams_index: Dict[str, Dict[str, Dict[str, Optional[str]]]]) -> List[Dict[str, Any]]:
    data = load_json(key_file)
    out = []
    for art in data:
        aid = art.get("id") or art.get("article_id") or art.get("doc_id")
        title = art.get("title", "")
        kws = art.get("keywords") or art.get("keyword") or []
        merged_kws = []
        for kw in kws:
            nk = normalize_kw(kw if isinstance(kw, str) else str(kw))
            desc = ""
            link = ""
            if aid and aid in bigrams_index:
                # only exact normalized match (no substring fallback)
                info = bigrams_index[aid].get(nk)
                if info:
                    desc = info.get("description") or ""
                    link = info.get("link") or ""
            merged_kws.append({"keyword": kw, "link": link, "description": desc})
        out.append({"id": aid, "title": title, "keywords": merged_kws})
    return out

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    train_bigrams = load_json(args.train_bigrams)
    val_bigrams = load_json(args.val_bigrams)
    test_bigrams = load_json(args.test_bigrams)

    train_idx = build_bigrams_index(train_bigrams)
    val_idx = build_bigrams_index(val_bigrams)
    test_idx = build_bigrams_index(test_bigrams)

    train_merged = merge_keywords_file(args.train, train_idx)
    val_merged = merge_keywords_file(args.val, val_idx)
    test_merged = merge_keywords_file(args.test, test_idx)

    grouped_path = os.path.join(args.out_dir, args.grouped_name)
    ordered_path = os.path.join(args.out_dir, args.ordered_name)

    grouped = {"train": train_merged, "val": val_merged, "test": test_merged}
    with open(grouped_path, "w", encoding="utf-8") as fh:
        json.dump(grouped, fh, indent=2, ensure_ascii=False)

    ordered = []
    ordered.append({"__section__": "train"})
    ordered.extend(train_merged)
    ordered.append({"__section__": "val"})
    ordered.extend(val_merged)
    ordered.append({"__section__": "test"})
    ordered.extend(test_merged)
    with open(ordered_path, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, indent=2, ensure_ascii=False)

    print("Wrote grouped file:", os.path.abspath(grouped_path))
    print("Wrote ordered file :", os.path.abspath(ordered_path))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="filter_train/filtered_train_keywords.json")
    p.add_argument("--val", default="filter_val/filtered_val_keywords.json")
    p.add_argument("--test", default="filter_test/filtered_test_keywords.json")
    p.add_argument("--train-bigrams", dest="train_bigrams", default="filter_train/filtered_train_bigrams.json")
    p.add_argument("--val-bigrams", dest="val_bigrams", default="filter_val/filtered_val_bigrams.json")
    p.add_argument("--test-bigrams", dest="test_bigrams", default="filter_test/filtered_test_bigrams.json")
    p.add_argument("--out-dir", default="combined")
    p.add_argument("--grouped-name", default="combined_keywords_grouped.json")
    p.add_argument("--ordered-name", default="combined_keywords_ordered.json")
    args = p.parse_args()
    main(args)