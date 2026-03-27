import json
import pickle
import os

def load_embeddings(embedding_path):
    with open(embedding_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # If it's a list, convert to dict
        if isinstance(data, list):
            return {item.get("concept_id", item.get("id", item.get("key"))): item["embedding"] for item in data}
        return data

def build_graph_from_umls(article, embeddings, article_id):
    cuis = set()
    nfeatures_dict = {}
    for match_group in article.get("matches", []):
        for match in match_group:
            cui = match["cui"]
            cuis.add(cui)
            emb_key = f"{article_id}:{cui}"
            # Inside build_graph_from_umls, before the emb_key check:
            print("Looking for emb_key:", emb_key)
            if emb_key in embeddings:
                nfeatures_dict[cui] = embeddings[emb_key]
    nodes = [cui for cui in cuis if cui in nfeatures_dict]  # Only keep nodes with embeddings
    nfeatures = [nfeatures_dict[cui] for cui in nodes]      # List of embeddings in node order
    edges = []
    for i, cui1 in enumerate(nodes):
        for cui2 in nodes[i+1:]:
            edges.append((cui1, "co_occurs_with", cui2))
    return {
        "id": article_id,
        "nodes": nodes,
        "edges": edges,
        "nfeatures": nfeatures
    }

splits = [
    ("train", "../../embeddings/embeddings_train_allenai-scibert_scivocab_uncased.json"),
    ("validation", "../../embeddings/embeddings_val_allenai-scibert_scivocab_uncased.json"),
    ("test", "../../embeddings/embeddings_test_allenai-scibert_scivocab_uncased.json")
]

for split, emb_path in splits:
    input_path = f"elife_umls_concepts_{split}.json"
    output_path = f"../../graphs/{split}_umls_graphs_with_emb.pkl"
    os.makedirs("graphs", exist_ok=True)
    embeddings = load_embeddings(emb_path)
    # After loading embeddings
    print("Sample embedding keys:", list(embeddings.keys())[:2])
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    out_graphs = []
    if split == "train":
        # Only keep articles that have at least one embedding
        valid_ids = set(k.split(":")[0] for k in embeddings.keys())
        articles = [a for a in articles if a.get("id") in valid_ids]
    for article in articles:
        article_id = article.get("id")
        graph = build_graph_from_umls(article, embeddings, article_id)
        # Extra safety: skip graphs with no nodes/features
        if not graph["nodes"] or not graph["nfeatures"]:
            continue
        out_graphs.append(graph)
    with open(output_path, "wb") as pf:
        pickle.dump(out_graphs, pf)
    print(f"Saved {len(out_graphs)} graphs to {output_path}")
print("First article:", articles[0])