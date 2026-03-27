import os, sys, json
import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils.utils_bart import get_processed_elife_data, load_train_config, postprocess_text  # add postprocess_text to utils_bart or reimplement here

if len(sys.argv) < 3:
    print("Usage: python eval_bart_test.py <config.json> <model_dir> [out_dir]")
    sys.exit(1)

config = load_train_config(sys.argv[1])
model_dir = sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(model_dir, "eval")
os.makedirs(out_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(device)
model.eval()

# use test split only
test_loader = get_processed_elife_data(config["data_files"]["test"], tokenizer, config, "test", shuffle=False)

rouge = evaluate.load("rouge")
all_preds, all_refs = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   num_beams=config.get("num_beams", 4),
                                   max_length=config.get("decoder_max_length", 256),
                                   min_length=config.get("min_length", 0),
                                   length_penalty=config.get("length_penalty", 1.0),
                                   no_repeat_ngram_size=config.get("no_repeat_ngram_size", 0),
                                  )
        # prepare labels
        labels = batch.get("labels")
        if labels is None:
            raise RuntimeError("Test batches have no 'labels' — adapt to load refs from raw files.")
        labels = labels.cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        preds = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # optional postprocessing if you use sentence tokenization
        try:
            preds, decoded_labels = postprocess_text(preds, decoded_labels)
        except Exception:
            pass

        rouge.add_batch(predictions=preds, references=decoded_labels)
        all_preds.extend(preds); all_refs.extend(decoded_labels)

res = rouge.compute(use_stemmer=True)
res = {k: round(v * 100, 4) for k, v in res.items()}
print("ROUGE:", res)

with open(os.path.join(out_dir, "predictions_new.jsonl"), "w", encoding="utf-8") as fout:
    for p, r in zip(all_preds, all_refs):
        fout.write(json.dumps({"pred": p, "ref": r}, ensure_ascii=False) + "\n")
with open(os.path.join(out_dir, "rouge_results_new.json"), "w") as f:
    json.dump(res, f, indent=2)

print("Saved results to", out_dir)
