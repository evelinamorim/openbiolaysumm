import os
import json

# List all JSON files in the current folder
for filename in os.listdir('.'):
    if filename.endswith('.json'):
        with open(filename, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        for article in articles:
            bad_edges = []
            if 'edges' in article and isinstance(article['edges'], list):
                for i, edge in enumerate(article['edges']):
                    missing_keys = [k for k in ['from', 'to', 'type', 'similarity'] if k not in edge]
                    if missing_keys:
                        bad_edges.append((i, missing_keys))
            if bad_edges:
                print(f"File: {filename} | Article: {article.get('article_id', 'N/A')} | Bad edges: {bad_edges}")