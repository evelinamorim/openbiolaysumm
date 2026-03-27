import pandas as pd
df = pd.read_csv("combined_preds_for_manual_check.csv")
print("Rows:", len(df))
print("Pred counts:\n", df['pred'].value_counts(dropna=False))
print("\nprob_pos stats:")
print(df['prob_pos'].describe())
# show top positives
print("\nTop 20 by prob_pos:")
print(df.sort_values('prob_pos', ascending=False).head(20)[['id','keyword','pred','pred_label','prob_pos']].to_string(index=False))
# low-confidence positives (pred==1 but prob small)
if 'pred' in df.columns:
    print("\nPred==1 with prob_pos<0.6 (count):", len(df[(df['pred']==1) & (df['prob_pos']<0.6)]))
# change display format
pd.options.display.float_format = '{:.6f}'.format
print("\nSample small probs:")
print(df.sort_values('prob_pos').head(10)[['id','keyword','pred','prob_pos']].to_string(index=False))