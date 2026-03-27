import pickle

for split in ["train", "val", "test"]:
    pkl_path = f"graph_sample_{split}_with_features.pkl"
    with open(pkl_path, "rb") as f:
        articles = pickle.load(f)
    count = 0
    for article in articles:
        edges = article.get("edges", [])
        if not edges:
            print(f"PKL: {split} | Article ID: {article.get('article_id', 'N/A')}")
            count += 1
    print(f"Total articles with empty or missing edges in {split}: {count}")