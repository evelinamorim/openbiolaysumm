"""
dbpedia_enrich.py
-----------------
Second stage of the YAKE pipeline. Reads the _bigrams.json produced by
yake_preprocess.py --skip-dbpedia and enriches each entry with DBpedia
descriptions. Run this from a login node (which has internet access).

Usage:
    python dbpedia_enrich.py --input-file <path/to/processed_*_bigrams.json>

The file is enriched in-place (with a .bak backup created first).
"""

import os
import sys
import json
import shutil
import argparse
import requests


# ---------------------------------------------------------------------------
# DBpedia helpers (mirrored from yake_preprocess.py)
# ---------------------------------------------------------------------------

def search_dbpedia(keyword):
    lookup_url = f"http://lookup.dbpedia.org/api/search?query={keyword}&format=JSON"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(lookup_url, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json().get("docs", [])
            if not results:
                print(f"  No DBpedia results for '{keyword}'.")
                return None, None
            first_result = results[0]
            main_url = first_result.get("resource", [None])[0]
            if not main_url:
                return None, None
            full_description = fetch_dbpedia_description(main_url)
            return main_url, full_description
        print(f"  DBpedia lookup failed for '{keyword}' (HTTP {response.status_code}).")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"  Error querying DBpedia for '{keyword}': {e}")
        return None, None


def fetch_dbpedia_description(resource_url):
    resource_name = resource_url.split("/")[-1]
    dbpedia_api_url = f"http://dbpedia.org/data/{resource_name}.json"
    try:
        response = requests.get(dbpedia_api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            entity_data = data.get(f"http://dbpedia.org/resource/{resource_name}", {})
            abstracts = entity_data.get("http://dbpedia.org/ontology/abstract", [])
            for abstract in abstracts:
                if abstract.get("lang") == "en":
                    return abstract.get("value", "No abstract available")
        return "No abstract available"
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching description for '{resource_url}': {e}")
        return "No abstract available"


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich(input_file):
    print(f"Loading: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Count how many entries still need enrichment
    pending = [e for e in data if not e.get("dbpedia")]
    print(f"Total entries: {len(data)} | Pending DBpedia enrichment: {len(pending)}")

    if not pending:
        print("All entries already enriched. Nothing to do.")
        return

    # Backup original file before modifying
    backup_path = input_file + ".bak"
    shutil.copy2(input_file, backup_path)
    print(f"Backup saved to: {backup_path}")

    enriched_count = 0
    for i, entry in enumerate(data):
        # Skip entries that were already enriched (resume support)
        if entry.get("dbpedia"):
            continue

        keywords = entry.get("biomedical_keywords", [])
        if not keywords:
            continue

        print(f"\n[{i+1}/{len(data)}] '{entry.get('title', 'Untitled')}'")
        print(f"  Keywords: {keywords}")

        dbpedia_results = {}
        for kw in keywords:
            link, description = search_dbpedia(kw)
            if link:
                dbpedia_results[kw] = {"link": link, "description": description}
                print(f"  ✓ {kw} → {link}")

        entry["dbpedia"] = dbpedia_results
        enriched_count += 1

        # Save after every article so progress is never lost
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\nDone. Enriched {enriched_count} entries.")
    print(f"Output saved to: {input_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DBpedia enrichment stage for the YAKE pipeline. "
                    "Run from a login node with internet access."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the _bigrams.json file produced by yake_preprocess.py --skip-dbpedia.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: file not found: {args.input_file}")
        sys.exit(1)

    enrich(args.input_file)