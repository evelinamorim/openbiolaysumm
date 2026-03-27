import requests
import json
import time
from tqdm import tqdm
import os

def fetch_dbpedia_description(resource_url, retries=2, delay=1.0):
    resource_name = resource_url.split("/")[-1]
    dbpedia_api_url = f"http://dbpedia.org/data/{resource_name}.json"

    for attempt in range(retries + 1):
        try:
            response = requests.get(dbpedia_api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()

                # Search all entity blocks for the English abstract
                for entity_data in data.values():
                    if "http://dbpedia.org/ontology/abstract" in entity_data:
                        for abstract in entity_data["http://dbpedia.org/ontology/abstract"]:
                            if abstract.get("lang") == "en":
                                return abstract.get("value", "No abstract available")

            # Optional delay before retry
            if attempt < retries:
                time.sleep(delay)

        except Exception as e:
            print(f"Error fetching abstract for {resource_url} (attempt {attempt + 1}): {e}")
            if attempt < retries:
                time.sleep(delay)

    print(f"No English abstract found for: {resource_url}")
    return "No abstract available"

# --- DBpedia lookup: return top 10 candidate URLs with descriptions ---
def search_dbpedia_all(keyword):
    lookup_url = f"http://lookup.dbpedia.org/api/search?query={keyword}&maxResults=10&format=JSON"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(lookup_url, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json().get("docs", [])
            candidates = []
            for result in results:
                url_list = result.get("resource", [])
                if url_list:
                    url = url_list[0]
                    desc = fetch_dbpedia_description(url)
                    candidates.append({"url": url, "description": desc})
                    time.sleep(0.2)  # Politeness delay
            return candidates
        return []
    except Exception as e:
        print(f"Lookup error for '{keyword}': {e}")
        return []

# --- Load and Save ---
def load_keywords(split):
    path = f"../../YakePreProcess/filter_{split}/filtered_{split}_keywords.json"
    with open(path) as f:
        return json.load(f)

def save_results(results, split):
    os.makedirs("DBpedia_Candidates", exist_ok=True)
    path = f"../../DBpedia_Candidates/{split}_dbpedia_candidates.json"

    def safe_text(text):
        if isinstance(text, str):
            return text.replace('\u0000', '').strip()
        return ""

    # Sanitize all descriptions
    for keyword, candidates in results.items():
        for candidate in candidates:
            candidate["description"] = safe_text(candidate.get("description", ""))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_results(split):
    path = f"../../DBpedia_Candidates/{split}_dbpedia_candidates.json"
    if not os.path.exists(path):
        print(f"No saved results file found for split '{split}'")
        return {}
    with open(path) as f:
        return json.load(f)

def retry_missing_abstracts(split):
    print(f"🔄 Checking for missing abstracts in {split}...")
    results = load_results(split)
    if not results:
        return

    updated = 0
    still_missing = {}

    for keyword, candidates in tqdm(results.items()):
        for candidate in candidates:
            if candidate["description"] == "No abstract available":
                new_desc = fetch_dbpedia_description(candidate["url"])
                if new_desc != "No abstract available":
                    candidate["description"] = new_desc
                    updated += 1
                else:
                    still_missing.setdefault(keyword, []).append(candidate["url"])
                time.sleep(0.2)  # avoid hammering

    save_results(results, split)

    # Save the links that still have no descriptions
    if still_missing:
        os.makedirs("../../DBpedia_Candidates", exist_ok=True)
        with open(f"../../DBpedia_Candidates/{split}_missing_descriptions.json", "w") as f:
            json.dump(still_missing, f, indent=2)
        print(f"\n{len(still_missing)} keywords still have missing abstracts. "
              f"Saved to DBpedia_Candidates/{split}_missing_descriptions.json")

    print(f"\nRetry complete. Updated {updated} missing abstracts.")

# --- Main ---
if __name__ == "__main__":
    split_map = {"1": "val", "2": "test", "3": "train"}
    print("Choose an option:\n"
          "1. Fetch new DBpedia descriptions\n"
          "2. Retry missing abstracts in existing file\n"
          "3. Exit")
    choice = input("Your choice number: ").strip()

    if choice == "3":
        print("Exiting.")
        exit()

    print("\nChoose the split:\n1. val\n2. test\n3. train")
    split_choice = input("Your split number: ").strip()
    split = split_map.get(split_choice, "val")

    if choice == "1":
        keyword_docs = load_keywords(split)

        # Extract all unique keywords
        all_keywords = set()
        for doc in keyword_docs:
            for kw in doc.get("keywords", []):
                cleaned = kw.strip()
                if cleaned:
                    all_keywords.add(cleaned)

        print(f"Fetching top 10 DBpedia descriptions for {len(all_keywords)} unique keywords...")
        results = {}
        for keyword in tqdm(all_keywords):
            candidates = search_dbpedia_all(keyword)
            results[keyword] = candidates

        save_results(results, split)
        print(f"\nDone. Saved results to DBpedia_Candidates/{split}_dbpedia_candidates.json")

    elif choice == "2":
        retry_missing_abstracts(split)

    else:
        print("Invalid choice. Exiting.")
