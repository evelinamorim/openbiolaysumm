"""
ollama_filter.py
----------------
Stage 2 of the YAKE+BART pipeline. Runs on a SLURM NODE (GPU required).

Reads the _bigrams.json produced by yake_preprocess.py, which contains all
keywords that had a DBpedia match. For each keyword, uses DeepSeek-R1-Distill-Qwen-7B
via HuggingFace Transformers to classify whether it is truly biomedical/biological,
using both the keyword itself and its DBpedia description as context.

Uses batched inference for efficiency on GPU.

Output file (written to --output-folder):
  processed_<n>_biomedical.json  — articles with only biomedical-confirmed
                                   keywords + their DBpedia descriptions

Usage:
    python ollama_filter.py --input-file <path/to/processed_*_bigrams.json>

The script supports resuming: already-processed article IDs are skipped.
"""

import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_local_model_path(hf_home):
    """Resolve the local snapshot path, avoiding any HuggingFace hub calls."""
    model_dir = os.path.join(
        hf_home,
        "models--" + MODEL_NAME.replace("/", "--"),
        "snapshots",
    )
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model not found locally at: {model_dir}")
    snapshots = os.listdir(model_dir)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in: {model_dir}")
    return os.path.join(model_dir, snapshots[0])


def load_model(hf_home):
    local_path = get_local_model_path(hf_home)
    print(f"Loading model from: {local_path}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    print("Tokenizer loaded.", flush=True)

    print("Loading model weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}", flush=True)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

def build_prompt(keyword, description):
    # Use DeepSeek chat template to get structured yes/no after </think>
    return (
        f"<|begin_of_sentence|><|User|>Is the following term related to the biomedical "
        f"or biological domain? Answer only yes or no.\n"
        f"Term: {keyword}\n"
        f"Description: {description}<|Assistant|>"
    )


def parse_answer(generated_text):
    """Extract yes/no from DeepSeek output, handling <think> reasoning chains."""
    text = generated_text.lower()
    # Look at text after </think> if present
    if "</think>" in text:
        text = text.split("</think>")[-1]
    yes_pos = text.find("yes")
    no_pos  = text.find("no")
    if yes_pos == -1 and no_pos == -1:
        return False  # default to not biomedical if unclear
    if yes_pos == -1:
        return False
    if no_pos == -1:
        return True
    return yes_pos < no_pos


def classify_keywords_batch(keywords_with_meta, tokenizer, model, batch_size=32):
    """
    Classify a list of (keyword, description) pairs in batches.
    Returns a list of booleans (True = biomedical).
    """
    prompts = [build_prompt(kw, meta.get("description", "")) for kw, meta in keywords_with_meta]
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,     # enough for <think>...</think> + yes/no
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            is_bio = parse_answer(generated)
            results.append(is_bio)
            print(f"  [{i+j+1}/{len(prompts)}] '{keywords_with_meta[i+j][0]}' → {'yes' if is_bio else 'no'}")

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return []


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def filter_biomedical(input_file, output_folder, batch_size, hf_home):
    print(f"Loading: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: expected a JSON list of articles.")
        sys.exit(1)

    base = os.path.basename(input_file).replace("_bigrams.json", "")
    output_path = os.path.join(output_folder, f"{base}_biomedical.json")

    all_biomedical = load_existing(output_path)
    done_ids = {e["id"] for e in all_biomedical}

    pending = [e for e in data if e.get("id") not in done_ids]
    print(f"Total articles : {len(data)}")
    print(f"Already done   : {len(done_ids)}")
    print(f"Remaining      : {len(pending)}\n")

    if not pending:
        print("All articles already processed.")
        return

    tokenizer, model = load_model(hf_home)

    for i, entry in enumerate(pending):
        article_id = entry.get("id", f"idx_{i}")
        title      = entry.get("title", "Untitled")
        dbpedia    = entry.get("dbpedia", {})

        print(f"\n[{i+1}/{len(pending)}] {title}")

        if not dbpedia:
            all_biomedical.append({
                "id": article_id,
                "title": title,
                "biomedical_keywords": {},
            })
            save_json(output_path, all_biomedical)
            done_ids.add(article_id)
            continue

        # Classify all keywords for this article in one batch
        keywords_with_meta = list(dbpedia.items())
        is_biomedical = classify_keywords_batch(keywords_with_meta, tokenizer, model, batch_size)

        biomedical_keywords = {
            kw: meta
            for (kw, meta), keep in zip(keywords_with_meta, is_biomedical)
            if keep
        }

        kept    = len(biomedical_keywords)
        removed = len(keywords_with_meta) - kept
        print(f"  Kept: {kept} | Removed: {removed}")

        all_biomedical.append({
            "id": article_id,
            "title": title,
            "biomedical_keywords": biomedical_keywords,
        })
        save_json(output_path, all_biomedical)
        done_ids.add(article_id)

    print(f"\nDone. Output saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 — DeepSeek-R1 biomedical classification via HuggingFace Transformers. "
            "Run on a SLURM NODE with GPU."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        default=os.environ.get("BIGRAMS_INPUT_FILE"),
        help="Path to the _bigrams.json produced by yake_preprocess.py.",
    )
    parser.add_argument(
        "--output-folder",
        default=os.environ.get(
            "OLLAMA_OUTPUT_DIR",
            "/projects/F202600026AIVLABDEUCALION/Biomedical_Summary_Enhanced/YakePreProcess/Files_pre_processed",
        ),
        help="Output folder for the filtered JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of keywords to classify per batch (default: 32).",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", "./hf_cache"),
        help="HuggingFace cache directory. Also settable via $HF_HOME.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: input file not found: {args.input_file}")
        sys.exit(1)

    os.makedirs(args.output_folder, exist_ok=True)
    filter_biomedical(args.input_file, args.output_folder, args.batch_size, args.hf_home)