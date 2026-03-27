import os
import json
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text  


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must contain 'text' and 'label' columns: {path}")
    X = df["text"].fillna("").astype(str).tolist()
    y = df["label"].astype(int).tolist()
    return df, X, y


def compute_and_save_metrics(df, X, y, pipe, out_path, prefix):
    preds = pipe.predict(X)
    try:
        probs = pipe.predict_proba(X)[:, 1]
    except Exception:
        probs = None

    p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, digits=4, zero_division=0)
    metrics = {"precision": float(p), "recall": float(r), "f1": float(f), "accuracy": float(acc)}

    metrics_path = os.path.join(out_path, f"xgb_{prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    preds_out = df.copy()
    preds_out["pred"] = preds
    if probs is not None:
        preds_out["pred_prob"] = probs
    preds_out.to_csv(os.path.join(out_path, f"xgb_{prefix}_predictions.csv"), index=False)

    return metrics, report


def train_and_evaluate(X_train, y_train, X_test, y_test, train_df, test_df, args, out_subdir, split_label):
    """Train, evaluate, and save results for a given split."""
    out_path = os.path.join(args.out_dir, out_subdir)
    os.makedirs(out_path, exist_ok=True)

    # ✅ Use sklearn's built-in stopword list
    custom_stopwords = text.ENGLISH_STOP_WORDS

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words=list(custom_stopwords),   # <-- Added stopword removal here
            max_features=args.max_features,
            ngram_range=tuple(map(int, args.ngram_range.split(","))),
            max_df=args.max_df,
            min_df=args.min_df
        )),
        ("clf", xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=args.n_jobs,
            random_state=args.seed
        ))
    ])

    print(f"\n--- Training on {len(X_train)} examples ({split_label}) ---")
    pipe.fit(X_train, y_train)

    model_path = os.path.join(out_path, args.model_name)
    joblib.dump(pipe, model_path)
    print(f"Saved trained model to {model_path}")

    results = {}

    print("Evaluating on training split...")
    train_metrics, _ = compute_and_save_metrics(train_df, X_train, y_train, pipe, out_path, f"{split_label}_train")
    results["train"] = train_metrics
    print("Train metrics:", train_metrics)

    print("Evaluating on test split...")
    test_metrics, _ = compute_and_save_metrics(test_df, X_test, y_test, pipe, out_path, f"{split_label}_test")
    results["test"] = test_metrics
    print("Test metrics:", test_metrics)

    with open(os.path.join(out_path, f"xgb_{split_label}_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return results


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== Running FIXED SPLIT evaluation ===")
    train_df, X_train, y_train = load_data(args.train)
    test_df, X_test, y_test = load_data(args.test)
    fixed_results = train_and_evaluate(
        X_train, y_train, X_test, y_test, train_df, test_df, args, "fixed_split", "fixed"
    )

    print("\n=== Running RANDOMIZED SPLIT evaluation ===")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    train_df_rand, test_df_rand = train_test_split(
        combined_df,
        test_size=0.2,
        stratify=combined_df["label"],
        random_state=args.seed
    )

    X_train_rand = train_df_rand["text"].fillna("").astype(str).tolist()
    y_train_rand = train_df_rand["label"].astype(int).tolist()
    X_test_rand = test_df_rand["text"].fillna("").astype(str).tolist()
    y_test_rand = test_df_rand["label"].astype(int).tolist()

    random_results = train_and_evaluate(
        X_train_rand, y_train_rand, X_test_rand, y_test_rand,
        train_df_rand, test_df_rand, args, "random_split", "random"
    )

    summary = {"fixed_split": fixed_results, "random_split": random_results}
    with open(os.path.join(args.out_dir, "xgb_all_results_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nAll done. Results saved in:", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train.csv")
    p.add_argument("--test", required=True, help="Path to test.csv (20% split)")
    p.add_argument("--out-dir", default="./models/xgb_tfidf", help="Output folder")
    p.add_argument("--model-name", dest="model_name", default="xgb_tfidf.joblib", help="Model filename")
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--ngram-range", default="2,2", help="ngram range as '1,2'")
    p.add_argument("--max-df", dest="max_df", type=float, default=0.95)
    p.add_argument("--min-df", dest="min_df", type=int, default=2)
    p.add_argument("--n-jobs", dest="n_jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
