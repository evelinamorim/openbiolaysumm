import pandas as pd
df = pd.read_csv("combined_preds_for_manual_check.csv")
TH = 0.0000001  # change threshold to try 0.1, 0.2, 0.3...
df['pred_thresh_{}'.format(int(TH*100))] = (df['prob_pos'] >= TH).astype(int)
df.to_csv("combined_preds_for_manual_check_with_thresh.csv", index=False)
print("Wrote with threshold", TH, "rows:", len(df), "positives:", df['pred_thresh_{}'.format(int(TH*100))].sum())