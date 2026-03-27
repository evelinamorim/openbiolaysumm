import pickle

print("What file to check?")
print("1. train")
print("2. val")
print("3. test")

choice = input("Número da tua escolha: ").strip()

split_map = {
    "1": "train",
    "2": "val",
    "3": "test"
}

if choice in split_map:
    split = split_map[choice]
    PKL_PATH = f"Created_graphs/graph_sample_{split}_with_features.pkl"
    try:
        with open(PKL_PATH, "rb") as f:
            graph = pickle.load(f)
        print("Conteúdo do ficheiro carregado com sucesso:")
    except FileNotFoundError:
        print(f"Ficheiro não encontrado: {PKL_PATH}")
        exit()
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o ficheiro: {e}")
        exit()
else:
    print("Escolha inválida.")
    exit()

print("\nGrafo carregado com sucesso!\n")
print("Chaves disponíveis no grafo:", list(graph.keys()))

for key in graph:
    print(f"\n--- {key} ---")
    print(f"Type: {type(graph[key])}")
    try:
        print(f"Length: {len(graph[key])}")
    except Exception:
        print("No length (not a list or dict)")

    # Show example content
    if isinstance(graph[key], list) and len(graph[key]) > 0:
        print(f"First element type: {type(graph[key][0])}")
        print(f"First element: {graph[key][0]}")
        if isinstance(graph[key][0], (list, dict)):
            print(f"First element length: {len(graph[key][0])}")
    elif isinstance(graph[key], dict):
        sample_keys = list(graph[key].keys())[:3]
        print(f"Sample keys: {sample_keys}")
        for k in sample_keys:
            print(f"graph['{key}'][{k}] = {graph[key][k]}")
    else:
        print(f"Value: {graph[key]}")

# Try to infer if there's an article_id mapping
if "article_id_to_idx" in graph:
    print("\nFound article_id_to_idx mapping!")
    print("Sample mapping:", list(graph["article_id_to_idx"].items())[:5])
else:
    print("\nNo article_id_to_idx mapping found.")

# Show a summary for nodes, edges, nfeatures if present
for field in ["nodes", "edges", "nfeatures"]:
    if field in graph:
        print(f"\nField '{field}':")
        print(f"  Type: {type(graph[field])}")
        print(f"  Length: {len(graph[field])}")
        if len(graph[field]) > 0:
            print(f"  First element type: {type(graph[field][0])}")
            print(f"  First element: {graph[field][0]}")
            if isinstance(graph[field][0], (list, dict)):
                print(f"  First element length: {len(graph[field][0])}")

print("\nDone.")
