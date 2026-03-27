import os, sys, json
import torch
from torch.optim import AdamW
import nltk
import evaluate
from accelerate import Accelerator
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler

from src.utils.utils_bart import get_processed_elife_data, load_train_config, update_config

config_path = sys.argv[1]
config = load_train_config(config_path)
os.makedirs(config['output_dir'], exist_ok=True)

ds = "../../DSplit"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
accelerator = Accelerator(mixed_precision="fp16")  # use fp16 to reduce GPU memory

tokenizer = AutoTokenizer.from_pretrained(config['model_str'])
print(f"[DEBUG] Tokenizer vocab size: {len(tokenizer)}")
metric = evaluate.load("rouge")

train_dataloader = get_processed_elife_data(config['data_files']['train'], tokenizer, config, "train", shuffle=True)
val_dataloader   = get_processed_elife_data(config['data_files']['val'], tokenizer, config, "val", shuffle=False)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(config['model_str'])
print(f"[DEBUG] Model config vocab_size: {model.config.vocab_size}")
print(f"[DEBUG] Model embedding num_embeddings: {model.model.shared.num_embeddings}")
print(f"[DEBUG] Tokenizer vocab size: {len(tokenizer)}")

if len(tokenizer) != model.model.shared.num_embeddings:
    print(f"[WARNING] MISMATCH! Tokenizer has {len(tokenizer)} tokens but model embedding has {model.model.shared.num_embeddings}")
    print(f"[WARNING] Resizing model embeddings to match tokenizer")
    model.resize_token_embeddings(len(tokenizer))
else:
    print(f"[OK] Tokenizer and model embeddings match")

# CRITICAL: Validate data AFTER model is loaded
print("[DEBUG] Validating all training data token IDs against model vocab...")
model_vocab_size = model.model.shared.num_embeddings
max_seen_input = 0
max_seen_label = 0
bad_batches = []

for i, batch in enumerate(train_dataloader):
    max_input_id = batch['input_ids'].max().item()
    labels_non_pad = batch['labels'][batch['labels'] != -100]
    max_label_id = labels_non_pad.max().item() if labels_non_pad.numel() > 0 else 0
    
    max_seen_input = max(max_seen_input, max_input_id)
    max_seen_label = max(max_seen_label, max_label_id)
    
    if max_input_id >= model_vocab_size or max_label_id >= model_vocab_size:
        bad_batches.append((i, max_input_id, max_label_id))

if bad_batches:
    print(f"[ERROR] Found {len(bad_batches)} batches with token IDs out of range!")
    for batch_idx, max_in, max_lab in bad_batches[:5]:  # Show first 5
        print(f"  Batch {batch_idx}: max_input={max_in}, max_label={max_lab}")
    print(f"[ERROR] Model vocab size: {model_vocab_size}")
    raise ValueError(f"Data contains token IDs >= {model_vocab_size}. Data needs to be regenerated with correct tokenizer!")

print(f"[OK] Data validation passed. Max token IDs: input={max_seen_input}, label={max_seen_label}, model_vocab={model_vocab_size}")

# generation settings
model.config.num_beams = config.get('num_beams', 4)
model.config.max_length = config.get('decoder_max_length', 256)
model.config.min_length = config.get('min_length', 100)
model.config.length_penalty = config.get('length_penalty', 2.0)
model.config.no_repeat_ngram_size = config.get('no_repeat_ngram_size', 3)

optimizer = AdamW(model.parameters(), lr=config['lr'])

train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, val_dataloader, model, optimizer
)

num_training_steps = config['num_epochs'] * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
progress_bar = tqdm(range(num_training_steps))

print("Training...")
best_rougeL = float('-inf')
for epoch in range(config['num_epochs']):
    model.train()
    for step, batch in enumerate(train_dataloader):
        if 'idx' in batch:
            del batch['idx']
        if 'global_attention_mask' in batch:
            del batch['global_attention_mask']
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)

    # evaluation
    model.eval()
    metric = evaluate.load("rouge")
    for step, batch in enumerate(val_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            labels = batch["labels"]

            generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = np.where(generated_tokens != -100, generated_tokens, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    result = metric.compute(use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    print(f"Epoch {epoch} metric result: {result}")

    # save every epoch (overwrite previous epoch folder on rerun)
    save_dir = f"{config['output_dir']}/{ds}_epoch_{epoch}"
    print(f"Saving model and tokenizer to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    update_config(save_dir, config)

print("Done.")