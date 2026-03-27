import json
import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load SciBERT
scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
if scibert_tokenizer.pad_token is None:
   scibert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(texts):
    encoded_input = scibert_tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=250).to(device)
    with torch.no_grad():
        model_output = scibert(**encoded_input)
    pooled = mean_pooling(model_output, encoded_input['attention_mask'])
    projector = nn.Linear(768, 50).to(device)
    return projector(pooled)

def process_graph(input_path, output_path):
    with open(input_path, "r") as f:
        graph_data = json.load(f)

    all_nodes = []
    all_embeddings = []
    node_texts = []

    for article in graph_data["nodes"]:
        aid = article["article_id"]
        for kw in article["keywords"]:
            concept_id = f"{aid}:{kw['term']}"
            all_nodes.append(concept_id)
            node_texts.append(kw['term'])

    print(f"\n Generating embeddings for {len(all_nodes)} nodes in {os.path.basename(input_path)}...")
    batch_size = 32
    for i in range(0, len(node_texts), batch_size):
        batch = node_texts[i:i+batch_size]
        emb = get_sentence_embeddings(batch)
        all_embeddings.extend(emb.cpu().tolist())

    filtered_edges = []
    for edge in graph_data["edges"]:
        if edge["from"] in all_nodes and edge["to"] in all_nodes:
            filtered_edges.append((edge["from"], edge["to"], edge["similarity"]))

    final_graph = {
        "id": graph_data.get("id", os.path.basename(input_path)),
        "nodes": all_nodes,
        "edges": filtered_edges,
        "nfeatures": all_embeddings
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(final_graph, f)

    print(f"Saved to {output_path}")


# === Prompt ===
print("Escolhe o split para gerar embeddings:")
print("1. train")
print("2. val")
print("3. test")
print("4. todos")

choice = input("Your choice: ").strip()

split_map = {
    "1": "train",
    "2": "val",
    "3": "test",
    "4": "all"
}

selected = split_map.get(choice)

if not selected:
    print("Invalid choice. Leaving.")
    exit()

splits = ["train", "val", "test"] if selected == "all" else [selected]

for split in splits:
    input_file = f"../../graphs/graph_sample_{split}.json"
    output_file = f"../../graph_construction/Created_graphs/graph_sample_{split}_with_features.pkl"

    if os.path.exists(input_file):
        process_graph(input_file, output_file)
    else:
        print(f"File no found: {input_file} — ignoring.")
