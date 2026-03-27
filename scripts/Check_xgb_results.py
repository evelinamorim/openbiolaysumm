# python
import joblib, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
pipe = joblib.load("models/xgb_downsample/fixed_split/xgb_tfidf.joblib")   # adjust path if needed
df = pd.read_csv("Dset_downsample/test.csv")
X = df["text"].fillna("").astype(str).tolist()
y = df["label"].astype(int).tolist()
preds = pipe.predict(X)
p,r,f,_ = precision_recall_fscore_support(y,preds,average="binary",zero_division=0)
print("precision, recall, f1, acc:", p, r, f, accuracy_score(y,preds))
print("confusion matrix:\n", confusion_matrix(y,preds))
print(classification_report(y,preds, digits=4))

dfp = pd.read_csv("models/xgb_downsample/fixed_split/xgb_fixed_test_predictions.csv")  # or scripts/models/... depending where file is
print("FP examples (pred=1,label=0):")
print(dfp[(dfp.pred==1)&(dfp.label==0)].head(5)[["id","text"]])
print("\nFN examples (pred=0,label=1):")
print(dfp[(dfp.pred==0)&(dfp.label==1)].head(5)[["id","text"]])

for p in ["Dset_downsample/train.csv","Dset_downsample/test.csv"]:
    df = pd.read_csv(p)
    print(p, "duplicate rows by text+label:", df.duplicated(subset=["text","label"]).sum())

# feature importance
vec = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]
feat_names = vec.get_feature_names_out()
imp = clf.feature_importances_
top = sorted(zip(imp, feat_names), reverse=True)[:30]
for score,fn in top:
    print(score, fn)