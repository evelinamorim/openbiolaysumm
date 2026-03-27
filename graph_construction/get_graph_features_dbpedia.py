import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import os
import dgl
from sklearn.decomposition import PCA
import numpy as np
import joblib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load SciBERT
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=250).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()

def process_graph(split, pca_model=None):
    graph_path_in = f"../graphs/graph_sample_{split}.json"
    graph_path_out = f"Created_graphs/graph_sample_{split}_with_features.pkl"

    if not os.path.exists(graph_path_in):
        print(f"Graph file not found: {graph_path_in}")
        return

    print(f"\n=== Processing split: {split} ===")

    # Load per-article graphs
    with open(graph_path_in, "r") as f:
        article_graphs = json.load(f)

    output_graphs = []

    # First pass: collect all node texts for PCA fitting (only for train split)
    if split == "train":
        all_node_texts = []
        for article in article_graphs:
            for kw in article["nodes"]:
                all_node_texts.append(kw["term"])
            for edge in article["edges"]:
                for node_id in [edge["from"], edge["to"]]:
                    all_node_texts.append(node_id.split(":", 1)[-1])
        # Remove duplicates
        all_node_texts = list(set(all_node_texts))
        print(f"Embedding {len(all_node_texts)} unique node texts for PCA fitting...")
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(all_node_texts), batch_size):
            batch_texts = all_node_texts[i:i+batch_size]
            batch_emb = embed_text(batch_texts)
            all_embeddings.append(batch_emb)
        all_embeddings = np.vstack(all_embeddings)
        # Fit PCA
        pca_model = PCA(n_components=50)
        pca_model.fit(all_embeddings)
        print("PCA fitted on train node embeddings.")
        # Save PCA model for later use
        joblib.dump(pca_model, "Created_graphs/pca_model.pkl")

    for article in article_graphs:
        article_id = article["article_id"]
        nodes = []
        node_texts = []

        # 1. Add all nodes from the article
        for kw in article["nodes"]:
            concept_id = f"{article_id}:{kw['term']}"
            nodes.append(concept_id)
            node_texts.append(kw["term"])

        # 2. Add any node referenced by an edge that isn't already in nodes
        for edge in article["edges"]:
            for node_id in [edge["from"], edge["to"]]:
                if node_id not in nodes:
                    nodes.append(node_id)
                    node_texts.append(node_id.split(":", 1)[-1])

        # 3. Generate SciBERT embeddings and reduce dimension with PCA
        node_embeddings = []
        batch_size = 32
        for i in range(0, len(node_texts), batch_size):
            batch_texts = node_texts[i:i+batch_size]
            batch_emb = embed_text(batch_texts)
            node_embeddings.append(batch_emb)
        node_embeddings = np.vstack(node_embeddings)  # shape: [num_nodes, 768]
        node_embeddings_reduced = pca_model.transform(node_embeddings)  # shape: [num_nodes, 50]

        nfeatures = node_embeddings_reduced.tolist()  # Use reduced embeddings as node features

        # 4. Filter edges to only those between valid nodes (now includes cross-article nodes)
        valid_nodes = set(nodes)
        filtered_edges = []
        for edge in article["edges"]:
            if edge["from"] in valid_nodes and edge["to"] in valid_nodes:
                filtered_edges.append((edge["from"], edge["type"], edge["to"], edge["similarity"]))

        output_graphs.append({
            "article_id": article_id,
            "nodes": nodes,
            "edges": filtered_edges,
            "nfeatures": nfeatures
        })

        print(f"Article {article['article_id']} nodes: {nodes}")

    os.makedirs(os.path.dirname(graph_path_out), exist_ok=True)
    with open(graph_path_out, "wb") as f:
        pickle.dump(output_graphs, f)

    # Print the shape of the node feature vector for the first article
    if len(output_graphs) > 0:
        print(f"Node feature vector dimension: {len(output_graphs[0]['nfeatures'][0])}")

    print(f"Saved to {graph_path_out}")

# === Run for all splits
for split in ["train", "val", "test"]:
    graph_path_out = f"Created_graphs/graph_sample_{split}_with_features.pkl"
    pca_path = "Created_graphs/pca_model.pkl"
    if split == "train":
        if os.path.exists(graph_path_out) and os.path.exists(pca_path):
            print(f"Skipping train split, {graph_path_out} and {pca_path} already exist.")
        else:
            process_graph(split)
    else:
        # Load PCA model fitted on train split
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model file not found: {pca_path}. Run train split first.")
        pca_model = joblib.load(pca_path)
        process_graph(split, pca_model)
