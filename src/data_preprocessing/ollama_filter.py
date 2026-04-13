"""
ollama_filter.py
----------------
Stage 2 of the YAKE+BART pipeline. Runs on a SLURM NODE (GPU required).

Reads the _bigrams.json produced by yake_preprocess.py, which contains all
keywords that had a DBpedia match. For each keyword, uses DeepSeek-R1 via
Ollama to classify whether it is truly biomedical/biological, using both the
keyword itself and its DBpedia description as context.

Output file (written to --output-folder):
  processed_<n>_biomedical.json  — articles with only biomedical-confirmed
                                   keywords + their DBpedia descriptions

Usage:
    python ollama_filter.py --input-file <path/to/processed_*_bigrams.json>

The script supports resuming: already-processed article IDs are skipped.

SLURM note: start Ollama on the compute node before running this script:
    ollama serve &
    sleep 10
    python ollama_filter.py --input-file ...
"""

import os
import sys
import json
import argparse
import requests


OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL   = "deepseek-r1:7b"


# ---------------------------------------------------------------------------
# Ollama classification
# ---------------------------------------------------------------------------

def query_ollama(keyword, description, ollama_url):
    """
    Ask DeepSeek-R1 whether a keyword + its DBpedia description are biomedical.
    Returns the raw response string, or None on error.
    """
    prompt = (
        f"You are a biomedical domain expert. Given the following term and its description, "
        f"classify whether it belongs to the biomedical or biological domain.\n\n"
        f"Term: {keyword}\n"
        f"Description: {description}\n\n"
        f"Answer with only 'yes' if it is biomedical/biological, or 'no' if it is not."
    )
    try:
        response = requests.post(
            ollama_url,
            headers={"Content-Type": "application/json"},
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            stream=True,
            timeout=60,
        )
        if response.status_code != 200:
            print(f"  Ollama error for '{keyword}' (HTTP {response.status_code}).")
            return None

        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    full_response += json.loads(chunk.decode("utf-8")).get("response", "")
                except json.JSONDecodeError:
                    pass

        result = full_response.strip()
        print(f"  Ollama → '{keyword}': {result[:80]}")
        return result

    except requests.exceptions.RequestException as e:
        print(f"  Error querying Ollama for '{keyword}': {e}")
        return None


def is_biomedical(keyword, description, ollama_url):
    result = query_ollama(keyword, description, ollama_url)
    if result:
        return "yes" in result.lower()
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing(path):
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

def filter_biomedical(input_file, output_folder, ollama_url):
    print(f"Loading: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: expected a JSON list of articles.")
        sys.exit(1)

    base = os.path.basename(input_file).replace("_bigrams.json", "")
    output_path = os.path.join(output_folder, f"{base}_biomedical.json")

    all_biomedical = load_existing(output_path)
    done_ids = {e["id"] for e in all_biomedical}

    print(f"Total articles in input : {len(data)}")
    print(f"Already processed       : {len(done_ids)}")
    print(f"Remaining               : {len(data) - len(done_ids)}\n")

    for i, entry in enumerate(data):
        article_id = entry.get("id", f"idx_{i}")
        title      = entry.get("title", "Untitled")
        dbpedia    = entry.get("dbpedia", {})

        if article_id in done_ids:
            continue

        if not dbpedia:
            # No DBpedia keywords to classify — save empty entry and move on
            all_biomedical.append({
                "id": article_id,
                "title": title,
                "biomedical_keywords": {},
            })
            save_json(output_path, all_biomedical)
            done_ids.add(article_id)
            continue

        print(f"[{i+1}/{len(data)}] {title}")

        biomedical_keywords = {}
        for keyword, meta in dbpedia.items():
            description = meta.get("description", "")
            if is_biomedical(keyword, description, ollama_url):
                biomedical_keywords[keyword] = meta
                print(f"  ✓ Kept   : {keyword}")
            else:
                print(f"  ✗ Removed: {keyword}")

        all_biomedical.append({
            "id": article_id,
            "title": title,
            "biomedical_keywords": biomedical_keywords,
        })
        save_json(output_path, all_biomedical)
        done_ids.add(article_id)

    print(f"\nDone. Output saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 — Ollama/DeepSeek biomedical classification of DBpedia keywords. "
            "Run on a SLURM NODE with GPU. Requires Ollama running locally."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        default=os.environ.get("BIGRAMS_INPUT_FILE"),
        help="Path to the _bigrams.json produced by yake_preprocess.py. "
             "Also settable via $BIGRAMS_INPUT_FILE.",
    )
    parser.add_argument(
        "--output-folder",
        default=os.environ.get(
            "OLLAMA_OUTPUT_DIR",
            "/projects/F202600026AIVLABDEUCALION/Biomedical_Summary_Enhanced/YakePreProcess/Files_pre_processed",
        ),
        help="Output folder for the filtered JSON file. Also settable via $OLLAMA_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_API_URL", OLLAMA_API_URL),
        help="Ollama API endpoint. Also settable via $OLLAMA_API_URL.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: input file not found: {args.input_file}")
        sys.exit(1)

    os.makedirs(args.output_folder, exist_ok=True)
    filter_biomedical(args.input_file, args.output_folder, args.ollama_url)