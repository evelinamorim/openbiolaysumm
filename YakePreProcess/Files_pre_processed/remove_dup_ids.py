import json

with open('processed_train_bigrams.json', 'r') as f:
    data = json.load(f)

unique = {}
for entry in data:
    unique[entry['id']] = entry  # This will keep the last occurrence

deduplicated_data = list(unique.values())

print(f"Reduced from {len(data)} to {len(deduplicated_data)} entries after removing duplicates.")

with open('processed_train_bigrams.json', 'w') as f:
    json.dump(deduplicated_data, f, indent=2)

















