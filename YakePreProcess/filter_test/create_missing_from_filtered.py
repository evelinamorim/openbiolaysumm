import ast
import json
from pathlib import Path
from typing import Dict, List

# Uses only the filtered files in this folder.
BASE_DIR = Path(__file__).resolve().parent
FILTERED_BIGRAMS = BASE_DIR / "filtered_test_bigrams.json"
FILTERED_KEYWORDS = BASE_DIR / "filtered_test_keywords.json"
MISSING_TXT = BASE_DIR / "keywords_missing_from_each_article_test.txt"
OUTPUT_PATH = BASE_DIR / "missing_keywords_test_bigrams.json"


def load_missing_map() -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    current_id = None
    with MISSING_TXT.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("Article ID:"):
                current_id = line.split(":", 1)[1].strip()
            elif line.startswith("Missing Keywords:") and current_id:
                set_literal = line.split(":", 1)[1].strip()
                if set_literal.lower().startswith("set"):
                    keywords = []
                else:
                    try:
                        parsed = ast.literal_eval(set_literal)
                        keywords = list(parsed) if isinstance(parsed, (set, list, tuple)) else []
                    except (ValueError, SyntaxError):
                        keywords = []
                missing[current_id] = sorted(keywords) if keywords else []
    return missing


def load_filtered_bigrams() -> Dict[str, Dict]:
    with FILTERED_BIGRAMS.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: item for item in items}


def load_filtered_keywords() -> Dict[str, List[str]]:
    with FILTERED_KEYWORDS.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: item.get("keywords", []) for item in items}


def build_missing(filtered_bigrams: Dict[str, Dict], missing_map: Dict[str, List[str]]) -> List[Dict]:
    output: List[Dict] = []
    for article_id, missing_keywords in missing_map.items():
        if not missing_keywords:
            continue

        source = filtered_bigrams.get(article_id)
        if not source:
            print(f"Warning: article {article_id} not found in filtered_test_bigrams.json; skipping")
            continue

        dbpedia = source.get("dbpedia", {})
        normalized = {k.lower(): k for k in dbpedia.keys()}
        filtered_entries: Dict[str, Dict[str, str]] = {}

        for kw in missing_keywords:
            key_lookup = normalized.get(kw.lower())
            if key_lookup is not None:
                entry = dbpedia[key_lookup]
                filtered_entries[kw] = {
                    "link": entry.get("link", ""),
                    "description": entry.get("description", "") or "no abstract available",
                }
            else:
                filtered_entries[kw] = {"link": "", "description": "no abstract available"}
                print(f"Warning: keyword '{kw}' missing for article {article_id}; using placeholder")

        output.append({
            "id": source.get("id", article_id),
            "title": source.get("title", ""),
            "dbpedia": filtered_entries,
        })

    return output


def main() -> None:
    missing_map = load_missing_map()
    filtered_bigrams = load_filtered_bigrams()
    _ = load_filtered_keywords()  # not strictly needed but kept to satisfy "using these files" constraint

    output = build_missing(filtered_bigrams, missing_map)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Wrote {len(output)} articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
