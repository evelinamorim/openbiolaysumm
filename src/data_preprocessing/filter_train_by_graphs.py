import json
import pickle

# Paths (edit if needed)
TRAIN_JSON = "train.json"
PKL_PATH = "graph_construction/Created_graphs/graph_sample_train_with_features.pkl"
OUTPUT_JSON = "train_filtered.json"

# Number of articles to keep (set to None for all, or any integer between 1 and 1582)
NUM_ARTICLES = 1582  # <-- Change this value as needed

# 1. Load article IDs from PKL
with open(PKL_PATH, "rb") as f:
    graphs = pickle.load(f)
pkl_article_ids = set(g["article_id"] for g in graphs)

print(f"Loaded {len(pkl_article_ids)} article IDs from PKL.")

# 2. Load train.json
with open(TRAIN_JSON, "r") as f:
    articles = json.load(f)
print(f"Loaded {len(articles)} articles from train.json.")

# 3. Filter articles
filtered_articles = [a for a in articles if a["id"] in pkl_article_ids]
print(f"Filtered down to {len(filtered_articles)} articles.")

# 4. Limit number of articles if NUM_ARTICLES is set
if NUM_ARTICLES is not None:
    filtered_articles = filtered_articles[:NUM_ARTICLES]
    print(f"Truncated to {len(filtered_articles)} articles.")

# 5. Save filtered list
with open(OUTPUT_JSON, "w") as f:
    json.dump(filtered_articles, f, indent=2)
print(f"Saved filtered articles to {OUTPUT_JSON}.")

# After loading articles and pkl_article_ids
sample_ids = [a["id"] for a in articles[:5]]
print("Sample article IDs from train.json:", sample_ids)
print("Sample article IDs from PKL:", list(pkl_article_ids)[:5])

# Check for intersection
intersection = set(a["id"] for a in articles) & pkl_article_ids
print(f"Number of matching IDs: {len(intersection)}")

if filtered_articles:
    print("First filtered article:", filtered_articles[0])