import json
import random
import os

# Input file paths
bigrams_path = "../Files_pre_processed/processed_train_bigrams.json"
keywords_path = "../Files_pre_processed/processed_train_keywords.json"

# Output folder
output_folder = "filter_train"
os.makedirs(output_folder, exist_ok=True)

# Load bigrams articles
with open(bigrams_path, "r", encoding="utf-8") as f:
    bigrams_data = json.load(f)

# Pick 50 random articles and collect their IDs (preserve order)
random_articles = random.sample(bigrams_data, 50)
selected_ids_in_order = [article["id"] for article in random_articles]

# Save filtered bigrams
with open(f"{output_folder}/filtered_train_bigrams.json", "w", encoding="utf-8") as f:
    json.dump(random_articles, f, indent=2)

# Load keywords articles and index by ID for fast lookup
with open(keywords_path, "r", encoding="utf-8") as f:
    keywords_data = json.load(f)
keywords_dict = {entry["id"]: entry for entry in keywords_data}

# Filter keywords to match the same IDs and order as random_articles
filtered_keywords = [keywords_dict[id_] for id_ in selected_ids_in_order if id_ in keywords_dict]

with open(f"{output_folder}/filtered_train_keywords.json", "w", encoding="utf-8") as f:
    json.dump(filtered_keywords, f, indent=2)

print("Saved 50 random articles and their keywords to filter_train/")