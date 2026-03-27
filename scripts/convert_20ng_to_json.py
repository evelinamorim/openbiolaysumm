import json
import os
from sklearn.datasets import fetch_20newsgroups

OUT_JSON = "20newsgroups_all.json"
OUT_JSONL = "20newsgroups_all.jsonl"

def main():
    data = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))
    rows = []
    for i, text in enumerate(data.data):
        rows.append({"id": f"20ng_{i}", "text": text, "label": 0, "source": "20newsgroups"})
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote", OUT_JSON, "and", OUT_JSONL)

if __name__ == "__main__":
    main()