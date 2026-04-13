"""
yake_preprocess.py
------------------
Stage 1 of the YAKE+BART pipeline. Runs on the LOGIN NODE (internet required).

For each article:
  1. YAKE extracts keyword candidates from the article body
  2. Each keyword is looked up in DBpedia
  3. Keywords that return a DBpedia result are kept, along with their description

Output files (written to --output-folder):
  processed_<name>_keywords.json  — all YAKE keywords per article (before DBpedia filter)
  processed_<name>_bigrams.json   — articles with DBpedia-confirmed keywords + descriptions

Usage:
    python yake_preprocess.py <start_id> <end_id> --input-file <path/to/elife.json>

The script supports resuming: already-processed article IDs are skipped if the
output files already exist.
"""

import os
import sys
import json
import argparse
import yake
import requests


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(text, num_terms=10, dedupLim=0.9):
    extractor = yake.KeywordExtractor(top=num_terms, n=2, dedupLim=dedupLim)
    return [kw[0] for kw in extractor.extract_keywords(text)]


# ---------------------------------------------------------------------------
# DBpedia
# ---------------------------------------------------------------------------

def search_dbpedia(keyword):
    """Look up a keyword in DBpedia. Returns (resource_url, description) or (None, None)."""
    lookup_url = f"http://lookup.dbpedia.org/api/search?query={keyword}&format=JSON"
    try:
        response = requests.get(lookup_url, headers={"Accept": "application/json"}, timeout=10)
        if response.status_code != 200:
            print(f"  DBpedia lookup failed for '{keyword}' (HTTP {response.status_code}).")
            return None, None

        results = response.json().get("docs", [])
        if not results:
            print(f"  No DBpedia results for '{keyword}'.")
            return None, None

        main_url = results[0].get("resource", [None])[0]
        if not main_url:
            return None, None

        description = fetch_dbpedia_description(main_url)
        return main_url, description

    except requests.exceptions.RequestException as e:
        print(f"  Error querying DBpedia for '{keyword}': {e}")
        return None, None


def fetch_dbpedia_description(resource_url):
    """Fetch the English abstract for a DBpedia resource URL."""
    resource_name = resource_url.split("/")[-1]
    api_url = f"http://dbpedia.org/data/{resource_name}.json"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            entity = data.get(f"http://dbpedia.org/resource/{resource_name}", {})
            for abstract in entity.get("http://dbpedia.org/ontology/abstract", []):
                if abstract.get("lang") == "en":
                    return abstract.get("value", "No abstract available")
        return "No abstract available"
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching description for '{resource_url}': {e}")
        return "No abstract available"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing(path):
    """Load a JSON list from path, returning an empty list on missing/corrupt file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return []


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_elife_file(file_path, output_folder, start_id, end_id):
    print(f"Processing: {file_path}  (articles {start_id}–{end_id})")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Skipping: JSON is not a list of articles.")
        return

    start_id = max(1, start_id)
    end_id = min(end_id, len(data))

    base = os.path.basename(file_path).replace(".json", "")
    keywords_path = os.path.join(output_folder, f"processed_{base}_keywords.json")
    bigrams_path  = os.path.join(output_folder, f"processed_{base}_bigrams.json")

    all_keywords = load_existing(keywords_path)
    all_bigrams  = load_existing(bigrams_path)

    # Build sets of already-processed IDs for resume support
    done_keywords = {e["id"] for e in all_keywords}
    done_bigrams  = {e["id"] for e in all_bigrams}

    processed = 0
    for index, article in enumerate(data[start_id - 1:end_id], start=start_id):
        if not isinstance(article, dict) or "sections" not in article:
            continue

        article_id = article.get("id", f"idx_{index}")
        title      = article.get("title", "Untitled")

        content = " ".join(
            " ".join(section)
            for section in article["sections"]
            if isinstance(section, list)
        ).strip()
        if not content:
            continue

        print(f"\n[{index}/{end_id}] {title}")

        # --- Stage 1a: YAKE keyword extraction ---
        if article_id not in done_keywords:
            keywords = extract_keywords(content)
            print(f"  YAKE keywords: {keywords}")
            all_keywords.append({"id": article_id, "title": title, "keywords": keywords})
            save_json(keywords_path, all_keywords)
            done_keywords.add(article_id)
        else:
            keywords = next(e["keywords"] for e in all_keywords if e["id"] == article_id)
            print(f"  YAKE keywords (cached): {keywords}")

        # --- Stage 1b: DBpedia lookup ---
        if article_id in done_bigrams:
            print("  DBpedia already done, skipping.")
            continue

        dbpedia_results = {}
        for kw in keywords:
            link, description = search_dbpedia(kw)
            if link:
                dbpedia_results[kw] = {"link": link, "description": description}
                print(f"  ✓ '{kw}' → {link}")

        all_bigrams.append({
            "id": article_id,
            "title": title,
            "dbpedia": dbpedia_results,
        })
        save_json(bigrams_path, all_bigrams)
        done_bigrams.add(article_id)

        processed += 1
        print(f"  Total processed: {processed}")

    print(f"\nDone.")
    print(f"  Keywords file : {keywords_path}")
    print(f"  DBpedia file  : {bigrams_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 — YAKE keyword extraction + DBpedia enrichment. "
            "Run from a LOGIN NODE (requires internet access)."
        )
    )
    parser.add_argument("start_id", type=int, help="First article index to process (1-based).")
    parser.add_argument("end_id",   type=int, help="Last article index to process (inclusive).")
    parser.add_argument(
        "--input-file",
        required=True,
        default=os.environ.get("ELIFE_INPUT_FILE"),
        help="Path to the eLife JSON file. Also settable via $ELIFE_INPUT_FILE.",
    )
    parser.add_argument(
        "--output-folder",
        default=os.environ.get(
            "YAKE_OUTPUT_DIR",
            "/projects/F202600026AIVLABDEUCALION/Biomedical_Summary_Enhanced/YakePreProcess/Files_pre_processed",
        ),
        help="Output folder for processed JSON files. Also settable via $YAKE_OUTPUT_DIR.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: input file not found: {args.input_file}")
        sys.exit(1)

    os.makedirs(args.output_folder, exist_ok=True)
    process_elife_file(args.input_file, args.output_folder, args.start_id, args.end_id)