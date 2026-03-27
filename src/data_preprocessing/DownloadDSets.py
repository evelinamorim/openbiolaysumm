import json
from sklearn.datasets import fetch_20newsgroups

out_path = "20newsgroups_all.jsonl"
data = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))

with open(out_path, "w", encoding="utf-8") as fout:
    for i, text in enumerate(data.data):
        rec = {
            "id": f"20ng_{i}",
            "text": text,
            "label": 0,
            "source": "20newsgroups"
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("Wrote", out_path)

