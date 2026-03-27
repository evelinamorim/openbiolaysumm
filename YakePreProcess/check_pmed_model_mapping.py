import os, json, argparse
from typing import List
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def predict_texts(model, tokenizer, texts, device='cpu', max_length=256, batch_size=64):
    all_logits = []
    model.to(device); model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k,v in enc.items()}
            out = model(**enc)
            logits = out.logits.cpu().numpy()
            all_logits.append(logits)
    return np.vstack(all_logits)

def main(args):
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    # normalize path and prefer local files when folder exists
    model_path = os.path.expanduser(args.model_dir).replace("\\", "/")
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    else:
        # fallback to HF hub repo id (internet required)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    cfg = model.config
    print("Model config id2label:", getattr(cfg, "id2label", None))
    print("Model config label2id:", getattr(cfg, "label2id", None))
    # quick sanity texts
    sanity = [
        "This study investigates gene expression changes in cancer cells and protein phosphorylation.",
        "This article discusses political elections and voter turnout in 2020.",
        "Protein binding and enzyme kinetics of kinase inhibitors.",
        "A new algorithm for sorting arrays.",
    ]
    logits = predict_texts(model, tokenizer, sanity, device=device, max_length=args.max_length, batch_size=8)
    probs = softmax(logits)
    for t, l, p in zip(sanity, logits, probs):
        print("\nTEXT:", t[:120])
        print("logits:", l)
        print("softmax:", p)
        print("pred_label_idx:", int(p.argmax()), "prob_pos_candidate:", float(p[1]) if p.shape[0]>1 else float(p[0]))
    # load combined CSV (if present) and show distribution
    if os.path.exists(args.combined_csv):
        df = pd.read_csv(args.combined_csv)
        texts = (df['keyword'].fillna("").astype(str) + ". " + df.get('description', "").fillna("").astype(str)).tolist()
        logits = predict_texts(model, tokenizer, texts, device=device, max_length=args.max_length, batch_size=args.batch_size)
        probs = softmax(logits)
        # decide positive index candidates
        print("\nCombined CSV size:", len(texts))
        if probs.shape[1] == 1:
            pos_prob = probs[:,0]
            pos_idx = 0
        else:
            pos_prob = probs[:,1]
            pos_idx = 1
        preds = (pos_prob >= args.threshold).astype(int)
        print("Using threshold", args.threshold, "-> pos index assumed:", pos_idx)
        print("Pred counts (by threshold):", dict(pd.Series(preds).value_counts()))
        # show top examples
        top = pd.DataFrame({'keyword': df['keyword'], 'description': df.get('description',""), 'prob_pos': pos_prob}).sort_values('prob_pos', ascending=False).head(20)
        print("\nTop 20 by pos-prob:")
        print(top.to_string(index=False))
    else:
        print("Combined CSV not found at:", args.combined_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--combined-csv", default="YakePreProcess/combined/combined_preds_for_manual_check.csv")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--use-cuda", action="store_true")
    args = p.parse_args()
    main(args)