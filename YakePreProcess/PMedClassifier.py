import os
import json
import argparse
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import numpy as np

def load_model(model_dir: str, device: str):
    try:
        # prefer local files when the path exists on disk
        model_path = os.path.expanduser(model_dir).replace("\\", "/")
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer from '{model_dir}': {e}")
    model.to(device)
    model.eval()
    return tokenizer, model

def batch_predict(texts: List[str], tokenizer, model, device: str, max_length: int, batch_size: int):
    """Return (preds:list[int], probs:list[float], logits:list[list[float]])"""
    preds = []
    probs = []
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.cpu().numpy()
            # softmax -> probabilities
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            soft = exp / exp.sum(axis=1, keepdims=True)
            pred = np.argmax(soft, axis=1)
            # choose positive class prob if binary else second column
            prob_pos = soft[:, 1] if soft.shape[1] > 1 else soft[:, 0]
        preds.extend(pred.tolist())
        probs.extend(prob_pos.tolist())
        all_logits.extend(logits.tolist())
    return preds, probs, all_logits

def main(args):
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    tokenizer, model = load_model(args.model_dir, device)

    # determine label mapping (use model.config.id2label if present)
    cfg = getattr(model, "config", None)
    id2label = getattr(cfg, "id2label", None)
    if id2label:
        print("Model id2label:", id2label)
    else:
        print("Model has no id2label in config. Will use --biomed-index to interpret labels.")
    biomed_index = args.biomed_index
    print(f"Interpreting label index {biomed_index} as 'biomedical' for outputs.")

    # read combined file (ordered or grouped)
    with open(args.input, "r", encoding="utf-8") as fh:
        combined = json.load(fh)

    # Detect if this is a dbpedia-format file (filter_train/val/test bigrams or merged files)
    is_dbpedia_format = (
        ("filter_train" in args.input or "filter_val" in args.input or "filter_test" in args.input) 
        and ("bigrams" in args.input or "merged" in args.input)
    )
    
    # If grouped (dict with train/val/test) convert to list preserving section markers
    sections = []
    if is_dbpedia_format:
        # Handle dbpedia format: list of articles with "dbpedia" dict containing keywords
        if isinstance(combined, list):
            sections = combined
        else:
            raise RuntimeError("Expected list of articles for dbpedia format")
    elif isinstance(combined, dict) and any(k in combined for k in ("train","val","test")):
        for sec in ("train","val","test"):
            if sec in combined:
                sections.append({"__section__": sec})
                sections.extend(combined[sec])
    elif isinstance(combined, list):
        sections = combined
    else:
        raise RuntimeError("Unsupported combined file format (expected dict with train/val/test or list)")

    rows_for_csv = []
    texts_to_run = []
    meta_entries = []  # keep pointers to where each text belongs

    # build per-keyword texts (use keyword + description for context)
    for entry in sections:
        if isinstance(entry, dict) and entry.get("__section__"):
            # section marker
            sec_marker = entry["__section__"]
            continue
        art_id = entry.get("id", "")
        title = entry.get("title", "")
        
        # Handle dbpedia format vs regular keywords format
        if is_dbpedia_format:
            # Keywords are in entry["dbpedia"] as a dict where keys are keyword names
            dbpedia_dict = entry.get("dbpedia", {}) or {}
            kws = []
            for keyword_name, keyword_data in dbpedia_dict.items():
                if isinstance(keyword_data, dict):
                    kws.append({
                        "keyword": keyword_name,
                        "description": keyword_data.get("description", ""),
                        "link": keyword_data.get("link", "")
                    })
        else:
            # Regular format: keywords are in entry["keywords"]
            kws = entry.get("keywords", []) or []
        
        for kw in kws:
            # each kw might be dict {'keyword': ..., 'description': ..., 'link': ...}
            if isinstance(kw, dict):
                ktext = kw.get("keyword") or kw.get("text") or ""
                desc = kw.get("description","") or ""
                klink = kw.get("link") or kw.get("url") or kw.get("href") or ""
            else:
                ktext = str(kw)
                desc = ""
                klink = ""

            # build input text with optional id/title and choice to use only keyword
            if args.only_keyword:
                parts = [f"Keyword: {ktext.strip()}"]
            else:
                parts = [f"Keyword: {ktext.strip()}"]
                if args.include_title and title:
                    parts.append(f"Title: {title.strip()}")
                if args.include_id and art_id:
                    parts.append(f"ArticleID: {art_id.strip()}")

                # description: included by default unless --no-description is set
                include_desc = not args.no_description
                if include_desc and desc and desc.strip():
                    parts.append(f"Context: {desc.strip()}")

                # link: only included if --include-link passed
                if args.include_link and klink:
                    parts.append(f"Link: {klink.strip()}")

            inp = " ".join(parts)
            texts_to_run.append(inp[: args.max_length*2])
            meta_entries.append({"id": art_id, "title": title, "keyword": ktext, "description": desc, "link": klink})
    if len(texts_to_run) == 0:
        print("No keywords found to classify.")
        return

    preds, probs, logits = batch_predict(texts_to_run, tokenizer, model, device, args.max_length, args.batch_size)

    if not (len(preds) == len(probs) == len(logits) == len(texts_to_run)):
        raise RuntimeError(f"Prediction length mismatch: preds={len(preds)} probs={len(probs)} logits={len(logits)} inputs={len(texts_to_run)}")

    # attach results and prepare outputs (same reconstruction) but include logits and write atomically
    out_json_entries = []
    idx = 0
    for entry in sections:
        if isinstance(entry, dict) and entry.get("__section__"):
            out_json_entries.append(entry)
            continue
        art = {"id": entry.get("id",""), "title": entry.get("title","")}
        
        # Handle dbpedia format vs regular keywords format for output
        if is_dbpedia_format:
            art["dbpedia"] = {}
            dbpedia_dict = entry.get("dbpedia", {}) or {}
            for keyword_name, keyword_data in dbpedia_dict.items():
                if isinstance(keyword_data, dict):
                    ktext = keyword_name
                    desc = keyword_data.get("description", "")
                else:
                    ktext = keyword_name
                    desc = ""
                
                pred = int(preds[idx])
                prob = float(probs[idx])
                logit = logits[idx]
                if id2label:
                    label_name = id2label.get(str(pred)) or id2label.get(pred) or str(pred)
                else:
                    label_name = "biomedical" if pred == biomed_index else "non-biomedical"

                art["dbpedia"][ktext] = {
                    "description": desc,
                    "link": keyword_data.get("link", "") if isinstance(keyword_data, dict) else "",
                    "pred": pred,
                    "pred_label": label_name,
                    "prob_pos": prob,
                    "logits": logit
                }
                rows_for_csv.append({
                    "id": art["id"],
                    "title": art["title"],
                    "keyword": ktext,
                    "description": desc,
                    "pred": pred,
                    "pred_label": label_name,
                    "prob_pos": prob
                })
                idx += 1
        else:
            # Regular format with keywords list
            art["keywords"] = []
            kws = entry.get("keywords",[]) or []
            for kw in kws:
                if isinstance(kw, dict):
                    ktext = kw.get("keyword") or kw.get("text") or ""
                    desc = kw.get("description","") or ""
                else:
                    ktext = str(kw)
                    desc = ""
                pred = int(preds[idx])
                prob = float(probs[idx])
                logit = logits[idx]
                if id2label:
                    label_name = id2label.get(str(pred)) or id2label.get(pred) or str(pred)
                else:
                    label_name = "biomedical" if pred == biomed_index else "non-biomedical"

                art["keywords"].append({
                    "keyword": ktext,
                    "description": desc,
                    "pred": pred,
                    "pred_label": label_name,
                    "prob_pos": prob,
                    "logits": logit
                })
                rows_for_csv.append({
                    "id": art["id"],
                    "title": art["title"],
                    "keyword": ktext,
                    "description": desc,
                    "pred": pred,
                    "pred_label": label_name,
                    "prob_pos": prob
                })
                idx += 1
        out_json_entries.append(art)

    # atomic write: tmp then replace
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    tmp_json = args.output_json + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as fh:
        json.dump(out_json_entries, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_json, args.output_json)

    df = pd.DataFrame(rows_for_csv)
    tmp_csv = args.output_csv + ".tmp"
    df.to_csv(tmp_csv, index=False, encoding="utf-8")
    os.replace(tmp_csv, args.output_csv)

    print("Wrote:", os.path.abspath(args.output_json))
    print("Wrote:", os.path.abspath(args.output_csv))
    print("Examples (top 10 by prob):")
    top = df.sort_values("prob_pos", ascending=False).head(10)
    print(top[["id","keyword","prob_pos","pred","pred_label"]].to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Folder with saved PubMedBERT model/tokenizer or HF id")
    p.add_argument("--input", default="YakePreProcess/combined/combined_keywords_ordered.json", help="Combined JSON file")
    p.add_argument("--output-json", default="YakePreProcess/combined/combined_with_preds.json", help="Augmented JSON output")
    p.add_argument("--output-csv", default="YakePreProcess/combined/combined_preds_for_manual_check.csv", help="CSV for manual checking")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--use-cuda", action="store_true", help="Use GPU if available")
    p.add_argument("--biomed-index", type=int, default=1, help="Index in model outputs to treat as 'biomedical' when id2label is absent")
    p.add_argument("--only-keyword", action="store_true", help="Classify using only the keyword text (ignore description/title)")
    p.add_argument("--include-title", action="store_true", help="Include article title in the classification context")
    p.add_argument("--include-id", action="store_true", help="Include article ID in the classification context")
    p.add_argument("--include-link", action="store_true", help="Include article link in the classification context")
    p.add_argument("--no-description", action="store_true", help="Exclude article description from the classification context")
    args = p.parse_args()
    main(args)