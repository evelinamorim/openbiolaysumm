import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Directory containing graph files
graphs_dir = "."

# List all JSON files in the graphs directory
graph_files = [f for f in os.listdir(graphs_dir) if f.endswith(".json")]

if not graph_files:
    print("No graph files found in the 'graphs' directory.")
    exit()

# Print menu of available graph files
print("Available graph files:")
for i, file_name in enumerate(graph_files, 1):
    print(f"{i}. {file_name}")

# Ask user to choose a graph file
choice = input(f"Select a graph to visualize (1-{len(graph_files)}): ")

try:
    choice_index = int(choice) - 1
    if choice_index < 0 or choice_index >= len(graph_files):
        raise ValueError
except ValueError:
    print("Invalid selection. Exiting.")
    exit()

selected_file = graph_files[choice_index]
file_path = os.path.join(graphs_dir, selected_file)

# Load graph from selected file
with open(file_path, "r") as f:
    data = json.load(f)

G = nx.Graph()

# Add nodes
for article in data["nodes"]:
    article_id = article["article_id"]
    for kw in article["keywords"]:
        node_id = f"{article_id}:{kw['term']}"
        G.add_node(node_id,
                   article_id=article_id,
                   term=kw['term'],
                   is_biomedical=(kw['is_biomedical'] == "yes"),
                   source=kw['source'])

# Add edges
for edge in data["edges"]:
    G.add_edge(edge["from"],
               edge["to"],
               similarity=edge["similarity"],
               type=edge["type"])

# Visual settings
pos = nx.spring_layout(G, seed=42)

# Biomedical = green, others = gray
node_colors = [
    'green' if G.nodes[n].get("is_biomedical") else 'lightgray'
    for n in G.nodes
]

# Highlight edges more clearly

# Normalize similarities for color mapping
similarities = [G.edges[e]['similarity'] for e in G.edges]
max_sim = max(similarities) if similarities else 1
min_sim = min(similarities) if similarities else 0

# Use a colormap (e.g., plasma) for edge colors based on similarity
edge_colors = [
    cm.plasma((G.edges[e]['similarity'] - min_sim) / (max_sim - min_sim + 1e-8))
    for e in G.edges
]

edge_widths = [max(G.edges[e]['similarity'] * 12, 3) for e in G.edges]  # even thicker

nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    edge_color=edge_colors,
    alpha=0.95
)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)

# Use only the keyword part (after the colon) for display
labels = {n: n.split(":", 1)[1] if ":" in n else n for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

# Title and save
plt.title(f"Biomedical Concept Similarity Graph - {selected_file}")
plt.axis("off")
plt.tight_layout()
png_filename = selected_file.rsplit('.', 1)[0] + ".png"
plt.savefig(png_filename, dpi=300)

print(f"Graph saved as '{png_filename}'")
