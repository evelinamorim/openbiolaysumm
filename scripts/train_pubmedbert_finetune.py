import os
import json
import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import torch
import random
import re

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def remove_source_signals(text):
    """Remove obvious domain/source indicators that could bias the model."""
    # Remove common source citations
    text = re.sub(r'(eLife|AG News|20newsgroups|Reuters|AP|SPACE\.com)', '', text, flags=re.IGNORECASE)
    # Remove copyright/attribution patterns
    text = re.sub(r'©.*?(?:\n|$)', '', text)
    text = re.sub(r'(By|From|--\s*)[A-Z][a-z]+.*?(?:\n|$)', lambda m: '\n' if '\n' in m.group(0) else '', text)
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text.strip()

def anonymize_id(id_str):
    """Remove source prefixes from IDs to prevent leakage."""
    # Remove common prefixes: elife_, ag_, 20ng_
    id_str = re.sub(r'^(elife|ag|20ng)_', '', str(id_str), flags=re.IGNORECASE)
    return id_str

def simple_augment(text):
    """Light text augmentation to prevent memorization of exact patterns."""
    # Randomly shuffle word order in non-critical parts (optional)
    # Or simply normalize the text slightly
    text = text.lower().strip()
    return text

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"precision": float(p), "recall": float(r), "f1": float(f), "accuracy": float(acc)}

def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # load train CSV
    df_train = pd.read_csv(args.train)
    if "text" not in df_train.columns or "label" not in df_train.columns:
        raise ValueError("train CSV must contain 'text' and 'label' columns")
    df_train["label"] = df_train["label"].astype(int)
    
    # Remove domain signals to force the model to learn deeper patterns
    if args.remove_domain_signals:
        print("Removing domain signals from text...")
        df_train["text"] = df_train["text"].apply(remove_source_signals)
        print("Anonymizing IDs...")
        df_train["id"] = df_train["id"].apply(anonymize_id)

    # create HF dataset
    ds_train = Dataset.from_pandas(df_train[["id","text","label"]].reset_index(drop=True))

    # tokenizer + model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_length)
    tokenized_train = ds_train.map(preprocess, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_steps=0,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # train
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # evaluate on training split (80%) and save metrics + predictions
    pred_train = trainer.predict(tokenized_train)
    train_metrics = compute_metrics(pred_train)
    # predictions array
    train_preds = np.argmax(pred_train.predictions, axis=1)
    probs = None
    try:
        probs = torch.softmax(torch.from_numpy(pred_train.predictions), dim=1)[:,1].numpy()
    except Exception:
        probs = None

    # save metrics and predictions
    with open(os.path.join(args.output_dir, "pubmedbert_train_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(train_metrics, fh, indent=2)

    df_out = df_train.copy().reset_index(drop=True)
    df_out["pred"] = train_preds
    if probs is not None:
        df_out["pred_prob"] = probs.tolist()
    df_out.to_csv(os.path.join(args.output_dir, "pubmedbert_train_predictions.csv"), index=False)

    # save summary
    summary = {"train": train_metrics}

    # --- New: evaluate on test split if provided ---
    if args.test:
        df_test = pd.read_csv(args.test)
        if "text" not in df_test.columns or "label" not in df_test.columns:
            raise ValueError("test CSV must contain 'text' and 'label' columns")
        df_test["label"] = df_test["label"].astype(int)
        
        # Apply same preprocessing to test data
        if args.remove_domain_signals:
            df_test["text"] = df_test["text"].apply(remove_source_signals)
            df_test["id"] = df_test["id"].apply(anonymize_id)
        
        ds_test = Dataset.from_pandas(df_test[["id","text","label"]].reset_index(drop=True))
        tokenized_test = ds_test.map(preprocess, batched=True, remove_columns=["text"])

        pred_test = trainer.predict(tokenized_test)
        test_metrics = compute_metrics(pred_test)
        test_preds = np.argmax(pred_test.predictions, axis=1)
        test_probs = None
        try:
            test_probs = torch.softmax(torch.from_numpy(pred_test.predictions), dim=1)[:,1].numpy()
        except Exception:
            test_probs = None

        with open(os.path.join(args.output_dir, "pubmedbert_test_metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(test_metrics, fh, indent=2)

        df_test_out = df_test.copy().reset_index(drop=True)
        df_test_out["pred"] = test_preds
        if test_probs is not None:
            df_test_out["pred_prob"] = test_probs.tolist()
        df_test_out.to_csv(os.path.join(args.output_dir, "pubmedbert_test_predictions.csv"), index=False)

        summary["test"] = test_metrics
        print("Test metrics:", test_metrics)
    # --- end test evaluation ---

    with open(os.path.join(args.output_dir, "pubmedbert_metrics_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # save training metadata
    meta = {"train_size": len(df_train), "model_name": model_name, "epochs": args.epochs, "batch_size": args.batch_size, "max_length": args.max_length}
    if args.test:
        meta["test_size"] = len(df_test)
    with open(os.path.join(args.output_dir, "train_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("Training finished. Results written to:", os.path.abspath(args.output_dir))
    print("Train metrics:", train_metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train.csv (80%% split)")
    p.add_argument("--test", required=False, help="Path to test.csv (20%% split) - optional; if provided the script will evaluate on test")
    p.add_argument("--output-dir", default="./models/pubmedbert", help="Output folder for model and metrics")
    p.add_argument("--model-name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", help="Hugging Face PubMedBERT model id or local folder")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--save-total-limit", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--remove-domain-signals", action="store_true", help="Remove source/domain indicators to force model to learn deeper patterns")
    args = p.parse_args()
    main(args)