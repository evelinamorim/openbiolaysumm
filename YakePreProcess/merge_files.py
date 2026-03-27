import json
import os
import argparse
from typing import Dict, List


def normalize_kw(s: str) -> str:
    """Normalize keyword for matching."""
    return " ".join(s.lower().strip().split())


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lookup(bigrams_data: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """
    Build:
      article_id -> normalized_keyword -> {description, link}
    """
    lookup = {}

    for art in bigrams_data:
        art_id = art.get("id")
        if not art_id:
            continue

        lookup[art_id] = {}
        dbpedia = art.get("dbpedia", {}) or {}

        for kw, meta in dbpedia.items():
            norm = normalize_kw(kw)
            if isinstance(meta, dict):
                lookup[art_id][norm] = {
                    "keyword": kw,
                    "description": meta.get("description", "") or "",
                    "link": meta.get("link", "") or ""
                }
            else:
                lookup[art_id][norm] = {
                    "keyword": kw,
                    "description": "",
                    "link": ""
                }

    return lookup


def merge_split(
    keywords_path: str,
    bigrams_path: str,
    output_path: str
):
    keywords_data = load_json(keywords_path)
    bigrams_data = load_json(bigrams_path)

    bigrams_lookup = build_lookup(bigrams_data)

    merged = []

    for art in keywords_data:
        art_id = art.get("id", "")
        title = art.get("title", "")
        yake_keywords = art.get("keywords", []) or []

        merged_art = {
            "id": art_id,
            "title": title,
            "dbpedia": {}
        }

        biomed_kw_lookup = bigrams_lookup.get(art_id, {})

        for kw in yake_keywords:
            kw_text = str(kw)
            norm = normalize_kw(kw_text)

            if norm in biomed_kw_lookup:
                meta = biomed_kw_lookup[norm]
                merged_art["dbpedia"][kw_text] = {
                    "description": meta["description"],
                    "link": meta["link"]
                }
            else:
                merged_art["dbpedia"][kw_text] = {
                    "description": "",
                    "link": ""
                }

        merged.append(merged_art)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✔ Wrote {output_path} ({len(merged)} articles)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Directory containing filtered_* files (for all splits)")
    parser.add_argument("--output-dir", required=True, help="Directory to write merged files")
    parser.add_argument("--split", help="Process only this split (train/val/test)")
    args = parser.parse_args()

    if args.split:
        # Process single split
        splits = [args.split]
        if args.input_dir:
            input_base = args.input_dir
        else:
            # Use split-specific directory: filter_train, filter_val, filter_test
            input_base = f"filter_{args.split}"
    else:
        # Process all splits
        splits = ["train", "val", "test"]
        if not args.input_dir:
            raise ValueError("--input-dir required when processing all splits")
        input_base = args.input_dir

    for split in splits:
        # If processing all splits, input_dir is the same
        # If processing single split, input_base already set correctly
        if args.split:
            input_dir = input_base
        else:
            input_dir = args.input_dir
            
        kw_path = os.path.join(input_dir, f"filtered_{split}_keywords.json")
        bg_path = os.path.join(input_dir, f"filtered_{split}_bigrams.json")
        out_path = os.path.join(args.output_dir, f"merged_{split}_dbpedia_all_keywords.json")

        if not os.path.exists(kw_path):
            print(f"⚠ Skipping {split}: {kw_path} not found")
            continue
        if not os.path.exists(bg_path):
            print(f"⚠ Skipping {split}: {bg_path} not found")
            continue

        merge_split(kw_path, bg_path, out_path)


if __name__ == "__main__":
    main()
