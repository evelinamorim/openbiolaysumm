import pandas as pd
import json

# Read the JSON file with dbpedia format
with open("train_preds.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Parse the dbpedia format into rows
rows = []
for article in data:
    art_id = article.get("id", "")
    title = article.get("title", "")
    dbpedia = article.get("dbpedia", {})
    
    for keyword, kw_data in dbpedia.items():
        if isinstance(kw_data, dict):
            rows.append({
                "id": art_id,
                "title": title,
                "keyword": keyword,
                "description": kw_data.get("description", ""),
                "pred": kw_data.get("pred", 0),
                "pred_label": kw_data.get("pred_label", ""),
                "prob_pos": kw_data.get("prob_pos", 0.0)
            })

df = pd.DataFrame(rows)

TH = 0.5  # change threshold to try 0.1, 0.2, 0.3...
df['pred_thresh_{}'.format(int(TH*100))] = (df['prob_pos'] >= TH).astype(int)
df.to_csv("train_preds_for_manual_check_with_thresh.csv", index=False)
print("Wrote with threshold", TH, "rows:", len(df), "positives:", df['pred_thresh_{}'.format(int(TH*100))].sum())