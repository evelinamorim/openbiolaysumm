import os
import json
import csv
import tarfile
import tempfile
from urllib.request import urlopen, urlretrieve

OUT_JSON = "ag_news_all.json"   # JSON array
OUT_JSONL = "ag_news_all.jsonl" # JSONL lines

def from_datasets():
    try:
        from datasets import load_dataset
    except Exception:
        return None
    ds = load_dataset("ag_news")
    rows = []
    for split in ("train","test"):
        for i, ex in enumerate(ds[split]):
            text = ex.get("text") or " ".join(filter(None, (ex.get("title",""), ex.get("description",""))))
            rows.append({"id": f"ag_{split}_{i}", "text": text, "label": 0, "source": "ag_news"})
    return rows

def from_tarball():
    url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv.tar.gz"
    tmp = tempfile.mkdtemp()
    tgz_path = os.path.join(tmp, "ag_news_csv.tar.gz")
    print("Downloading AG News tarball...")
    urlretrieve(url, tgz_path)
    rows = []
    with tarfile.open(tgz_path, "r:gz") as tf:
        for name in ("ag_news_csv/train.csv","ag_news_csv/test.csv"):
            try:
                member = tf.getmember(name)
            except KeyError:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            rdr = csv.reader(line.decode("utf-8", "replace") for line in f)
            for i, cols in enumerate(rdr):
                # format in this dataset: [label, title, description]
                if len(cols) >= 3:
                    _, title, desc = cols[0], cols[1], cols[2]
                    text = (title or "") + " " + (desc or "")
                else:
                    text = " ".join(cols)
                split = "train" if "train.csv" in name else "test"
                rows.append({"id": f"ag_{split}_{len(rows)}", "text": text, "label": 0, "source": "ag_news"})
    return rows

def save(rows):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote", OUT_JSON, "and", OUT_JSONL)

if __name__ == "__main__":
    os.makedirs("scripts", exist_ok=True)
    rows = from_datasets()
    if rows is None:
        rows = from_tarball()
    save(rows)