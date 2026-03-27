import os
import json

splits = ["val", "test", "train"]
output_dir = "FirstDBDescription"
os.makedirs(output_dir, exist_ok=True)

for split in splits:
    fname = f"{split}_dbpedia_candidates.json"
    output = {}
    if not os.path.isfile(fname):
        print(f"File not found: {fname}")
        continue
    with open(fname, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
        # data is a dict: {keyword: [candidates]}
        for keyword, candidates in data.items():
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                desc = first.get("description", "")
                if isinstance(keyword, str) and isinstance(desc, str) and keyword.strip() and desc.strip() and desc != "No abstract available":
                    output[keyword.strip()] = desc.strip()
    outname = os.path.join(output_dir, f"first_descriptions_{split}.txt")
    with open(outname, "w", encoding="utf-8") as f:
        for keyword, desc in output.items():
            f.write(f"{keyword}:{desc}\n")
    print(f"Saved {len(output)} keyword:description pairs to {outname}")