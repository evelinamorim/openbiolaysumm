# python
import json, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

m = json.load(open("xgb_test_metrics.json"))
print("Saved test metrics:", m)

df = pd.read_csv("xgb_test_predictions.csv")
y_true = df["label"]
y_pred = df["pred"]
print("Counts:", y_true.value_counts().to_dict())
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))
# show some example errors
print("\nExamples of false positives (pred=1, label=0):\n", df[(df.pred==1)&(df.label==0)].head(5)[["id","text"]])
print("\nExamples of false negatives (pred=0, label=1):\n", df[(df.pred==0)&(df.label==1)].head(5)[["id","text"]])