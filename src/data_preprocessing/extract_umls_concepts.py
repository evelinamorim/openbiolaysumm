from quickumls import QuickUMLS
import os
import json
import time

def build_cui_to_name_dict(mrconso_path):
    cui_to_name = {}
    with open(mrconso_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) > 14:
                cui, lang, ispref, name = parts[0], parts[1], parts[6], parts[14]
                if lang == "ENG" and ispref == "Y":
                    cui_to_name[cui] = name
    return cui_to_name

# Path to your QuickUMLS installation (where UMLS data is indexed)
quickumls_fp = "/projects/F202500013AIVLABDEUCALION/Biomedical_Summary_Enhanced/QuickUMLS_index"  # update this to your actual path

# Path to MRCONSO.RRF
mrconso_path = "/projects/F202500013AIVLABDEUCALION/Biomedical_Summary_Enhanced/2025AA/META/MRCONSO.RRF"

# Menu for selecting the split
split_files = {
    "1": ("Train", "train_filtered.json"),
    "2": ("Validation", "val.json"),
    "3": ("Test", "test.json"),
}

print("Select the split to process:")
for key, (name, _) in split_files.items():
    print(f"{key}: {name}")

choice = input("Enter the number of the split: ").strip()
if choice not in split_files:
    print("Invalid choice. Exiting.")
    exit(1)

split_name, elife_json_path = split_files[choice]

# Initialize QuickUMLS matcher
matcher = QuickUMLS(quickumls_fp, threshold=0.7, similarity_name="jaccard", window=5)

# Load elife articles
with open(elife_json_path, "r", encoding="utf-8") as f:
    articles = json.load(f)  # articles should be a list of dicts with a 'text' field

results = []

start_time = time.time()  # Start timer

# Build CUI to preferred name dictionary ONCE
cui_to_name = build_cui_to_name_dict(mrconso_path)

for article in articles:
    # Only use abstract and summary fields
    abstract = " ".join(article.get("abstract", []))
    summary = " ".join(article.get("summary", []))
    text = " ".join([abstract, summary]).strip()
    matches = matcher.match(text)
    # Flatten the list of lists into a single list of dicts
    flat_matches = [item for sublist in matches for item in sublist]
    # Optionally, keep only relevant fields for GoldSack compatibility
    formatted_matches = []
    for m in flat_matches:
        if "cui" in m:
            cui = m.get("cui")
            #if cui not in cui_to_name:
            #    print(f"CUI {cui} not found in MRCONSO.RRF dictionary!")
            #else:
            #    print(f"CUI {cui} maps to preferred name: {cui_to_name[cui]}")
            formatted_matches.append({
                "cui": cui,
                "term": m.get("term"),
                "similarity": m.get("similarity"),
                "preferred_name": cui_to_name.get(cui)
            })
    results.append({
        "id": article.get("id", None),
        "matches": formatted_matches
    })

# Save results to a file named for the split
output_path = f"DSplit/elife_umls_concepts_{split_name.lower()}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

elapsed = time.time() - start_time
print(f"Extraction complete. Results saved to {output_path}.")
print(f"Time taken for {split_name} split: {elapsed:.2f} seconds.")
