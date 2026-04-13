import os
import json
import sys
import argparse
import yake
import requests

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:7b"

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(text, num_terms=10, dedupLim=0.9):
    keyword_extractor = yake.KeywordExtractor(top=num_terms, n=2, dedupLim=dedupLim)
    keywords = [kw[0] for kw in keyword_extractor.extract_keywords(text)]
    return keywords

# ---------------------------------------------------------------------------
# DBpedia lookup
# ---------------------------------------------------------------------------

def search_dbpedia(keyword):
    lookup_url = f"http://lookup.dbpedia.org/api/search?query={keyword}&format=JSON"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(lookup_url, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json().get("docs", [])
            if not results:
                print(f"No results found in DBpedia for '{keyword}'.")
                return None, None
            first_result = results[0]
            main_url = first_result.get("resource", [None])[0]
            if not main_url:
                print("No valid DBpedia resource found.")
                return None, None
            print(f"First result URL: {main_url}")
            full_description = fetch_dbpedia_description(main_url)
            return main_url, full_description
        print(f"No relevant DBpedia link found for '{keyword}' (HTTP {response.status_code}).")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error querying DBpedia for '{keyword}': {e}")
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
        print(f"Error fetching description for '{resource_url}': {e}")
        return "No abstract available"

# ---------------------------------------------------------------------------
# Ollama / biomedical classification
# ---------------------------------------------------------------------------

def query_ollama(keyword):
    try:
        response = requests.post(
            OLLAMA_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": OLLAMA_MODEL,
                "prompt": (
                    f"Classify if the following term is related to the biomedical/biological domain: "
                    f"{keyword}. Answer 'yes' if it is related to biomedical/biological, 'no' if not."
                ),
            },
            stream=True,
            timeout=60,
        )

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from Ollama.")
            return None

        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    json_chunk = json.loads(chunk.decode("utf-8"))
                    full_response += json_chunk.get("response", "")
                except json.JSONDecodeError:
                    print("Error decoding chunk from Ollama.")

        print(f"Ollama response for '{keyword}': {full_response.strip()}")
        print("=" * 50)
        return full_response.strip()
    except requests.exceptions.RequestException as e:
        print(f"\nError querying Ollama: {e}")
        return None


def is_biomedical_keyword(keyword):
    result = query_ollama(keyword)
    if result:
        return "yes" in result.lower()
    return False

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_elife_file(file_path, output_folder, start_id, end_id, skip_dbpedia=False):
    print(f"Processing file: {file_path} (Articles {start_id} to {end_id})")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        print(f"Skipping {file_path}: JSON structure is not a list of articles.")
        return

    total_articles = len(data)
    start_id = max(1, start_id)
    end_id = min(end_id, total_articles)
    contador = 0

    base_filename = os.path.basename(file_path).replace(".json", "")
    output_filename = os.path.join(output_folder, f"processed_{base_filename}_bigrams.json")
    output_filename_keywords = os.path.join(output_folder, f"processed_{base_filename}_keywords.json")

    # Load existing data if files already exist (resume support)
    all_bigrams = []
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            try:
                all_bigrams = json.load(f)
            except json.JSONDecodeError:
                all_bigrams = []

    all_keywords = []
    if os.path.exists(output_filename_keywords):
        with open(output_filename_keywords, "r", encoding="utf-8") as f:
            try:
                all_keywords = json.load(f)
            except json.JSONDecodeError:
                all_keywords = []

    for index, article in enumerate(data[start_id - 1:end_id], start=start_id):
        if not isinstance(article, dict) or "sections" not in article:
            continue

        content = " ".join(
            " ".join(section) for section in article["sections"] if isinstance(section, list)
        ).strip()
        if not content:
            continue

        print(f"\n[Article {index}] Extracting keywords...")
        keywords = extract_keywords(content)
        print(f"Extracted keywords: {keywords}")

        processed_keywords_entry = {
            "id": article.get("id", "unknown"),
            "title": article.get("title", "Untitled"),
            "keywords": keywords,
        }
        all_keywords.append(processed_keywords_entry)
        with open(output_filename_keywords, "w", encoding="utf-8") as f:
            json.dump(all_keywords, f, ensure_ascii=False, indent=4)

        relevant_keywords = [kw for kw in keywords if is_biomedical_keyword(kw)]
        print(f"Filtered biomedical keywords: {relevant_keywords}")

        contador += 1
        print(f"Processed articles count: {contador}")

        if skip_dbpedia:
            # Save biomedical keywords only — DBpedia enrichment deferred to dbpedia_enrich.py
            processed_entry = {
                "id": article.get("id", "unknown"),
                "title": article.get("title", "Untitled"),
                "biomedical_keywords": relevant_keywords,
                "dbpedia": {},
            }
        else:
            dbpedia_results = {}
            for kw in relevant_keywords:
                link, description = search_dbpedia(kw)
                if link:
                    dbpedia_results[kw] = {"link": link, "description": description}
            print("DBpedia lookup complete for this article.")
            processed_entry = {
                "id": article.get("id", "unknown"),
                "title": article.get("title", "Untitled"),
                "biomedical_keywords": relevant_keywords,
                "dbpedia": dbpedia_results,
            }

        all_bigrams.append(processed_entry)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(all_bigrams, f, ensure_ascii=False, indent=4)

    print(f"\nDone. Results saved to:\n  {output_filename}\n  {output_filename_keywords}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="YAKE keyword extraction + DBpedia enrichment for biomedical lay summarization."
    )
    parser.add_argument("start_id", type=int, help="First article index to process (1-based).")
    parser.add_argument("end_id", type=int, help="Last article index to process (inclusive).")
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the eLife JSON file to process (e.g. /path/to/elife_train.json). "
             "Can also be set via the ELIFE_INPUT_FILE environment variable.",
        default=os.environ.get("ELIFE_INPUT_FILE"),
    )
    parser.add_argument(
        "--output-folder",
        default=os.environ.get(
            "YAKE_OUTPUT_DIR",
            "/projects/F202600026AIVLABDEUCALION/Biomedical_Summary_Enhanced/YakePreProcess/Files_pre_processed",
        ),
        help="Folder where processed output JSON files will be saved. "
             "Can also be set via the YAKE_OUTPUT_DIR environment variable.",
    )
    parser.add_argument(
        "--skip-dbpedia",
        action="store_true",
        help="Skip DBpedia enrichment (use on SLURM nodes without internet access). "
             "Run dbpedia_enrich.py afterwards from a login node to complete enrichment.",
        default=os.environ.get("OLLAMA_API_URL", OLLAMA_API_URL),
        help="Ollama API base URL (default: http://127.0.0.1:11434/api/generate). "
             "Can also be set via the OLLAMA_API_URL environment variable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: input file not found: {args.input_file}")
        sys.exit(1)

    # Allow overriding the Ollama URL at runtime
    OLLAMA_API_URL = args.ollama_url

    os.makedirs(args.output_folder, exist_ok=True)

    process_elife_file(args.input_file, args.output_folder, args.start_id, args.end_id,
                       skip_dbpedia=args.skip_dbpedia)