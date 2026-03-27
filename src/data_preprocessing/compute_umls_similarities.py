import json
import os
from sentence_transformers import SentenceTransformer
import torch
from torch.nn.functional import cosine_similarity

# Step 1: Prompt user for model selection
print("Select the model to encode the descriptions:")
print("1. all-MiniLM-L6-v2 (default, lightweight and fast)")
print("2. paraphrase-MiniLM-L6-v2 (optimized for paraphrase detection)")
print("3. all-mpnet-base-v2 (more accurate but slower)")
print("4. paraphrase-mpnet-base-v2 (optimized for paraphrase detection)")
print("5. distiluse-base-multilingual-cased-v2 (multilingual support)")
print("6. allenai/scibert_scivocab_uncased (SciBERT, optimized for scientific text)")
model_choice = input("Enter the number corresponding to your choice: ")

# Map user input to model names
model_map = {
    "1": "all-MiniLM-L6-v2",
    "2": "paraphrase-MiniLM-L6-v2",
    "3": "all-mpnet-base-v2",
    "4": "paraphrase-mpnet-base-v2",
    "5": "distiluse-base-multilingual-cased-v2",
    "6": "allenai/scibert_scivocab_uncased" 
}

# Get the selected model or default to 'all-MiniLM-L6-v2'
model_name = model_map.get(model_choice, "all-MiniLM-L6-v2")
model_tag = model_name.replace("/", "-")  # Safe for filenames
print(f"Using model: {model_name}")

# Step 2: Load model
model = SentenceTransformer(model_name)

# Step 3: Ask user for which split to use
print("Which split do you want to process?")
print("1. val")
print("2. test")
print("3. train")
print("4. val_50_random")
print("5. test_50_random")
print("6. train_50_random") 
split_choice = input("Enter the number corresponding to your choice: ").strip()
split_map = {
    "1": "val",
    "2": "test",
    "3": "train",
    "4": "val_50_random",
    "5": "test_50_random",
    "6": "train_50_random"
}
split = split_map.get(split_choice)
if not split:
    print("Invalid choice. Exiting.")
    exit()

file_info = {
    "val": "elife_umls_concepts_validation.json",
    "test": "elife_umls_concepts_test.json",
    "train": "elife_umls_concepts_train.json"
    # (add random splits if you have them)
}

os.makedirs("embeddings", exist_ok=True)
os.makedirs("similarities", exist_ok=True)

# Step 4: Function to compute highest and lowest cosine similarity per concept
def compute_extremes(concept_ids, embeddings):
    results = {cid: {"most_similar": None, "least_similar": None, "max_val": -1.0, "min_val": 1.0} for cid in concept_ids}
    num = len(concept_ids)

    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            sim = cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()

            cid_i = concept_ids[i]
            cid_j = concept_ids[j]

            # Update max similarity
            if sim > results[cid_i]["max_val"]:
                results[cid_i]["max_val"] = sim
                results[cid_i]["most_similar"] = {"concept_id": cid_j, "similarity": sim}

            # Update min similarity
            if sim < results[cid_i]["min_val"]:
                results[cid_i]["min_val"] = sim
                results[cid_i]["least_similar"] = {"concept_id": cid_j, "similarity": sim}

    # Format for JSON
    formatted = []
    for cid, data in results.items():
        formatted.append({
            "concept_id": cid,
            "most_similar": data["most_similar"],
            "least_similar": data["least_similar"]
        })
    return formatted

# Step 5: Process the selected dataset
label = split
path = file_info[label]
descriptions = []
concept_ids = []

with open(path, 'r', encoding='utf-8') as f:
    articles = json.load(f)

for article in articles:
    article_id = article["id"]
    for match_group in article.get("matches", []):
        for match in match_group:
            cui = match["cui"]
            desc = match.get("ngram") or match.get("term") or ""
            if desc:
                descriptions.append(desc)
                concept_ids.append(f"{article_id}:{cui}")

print(f"Loaded {len(descriptions)} descriptions from {label} set.")

# Encode descriptions
embeddings = model.encode(descriptions, convert_to_tensor=True)

# Save embeddings to JSON
emb_out = [
    {"concept_id": cid, "embedding": emb.tolist()}
    for cid, emb in zip(concept_ids, embeddings)
]
emb_filename = f"embeddings/embeddings_{label}_{model_tag}.json"
with open(emb_filename, "w", encoding="utf-8") as f:
    json.dump(emb_out, f, indent=2)
print(f"Saved embeddings to {emb_filename}")

# Compute and save similarity extremes
similarities = compute_extremes(concept_ids, embeddings)
sim_filename = f"similarities/similarities_{label}_{model_tag}.json"
with open(sim_filename, "w", encoding="utf-8") as f:
    json.dump(similarities, f, indent=2)
print(f"Saved extreme similarity pairs to {sim_filename}")
