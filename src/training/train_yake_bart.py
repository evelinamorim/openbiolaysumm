import os, sys, json
import torch
from torch.optim import AdamW
import nltk
import evaluate
from accelerate import Accelerator
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler

from src.utils.utils_yake_bart import get_processed_elife_data, load_train_config, update_config

config_path = sys.argv[1]
config = load_train_config(config_path)
os.makedirs(config['output_dir'], exist_ok=True)

ds = "checkpoints"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
accelerator = Accelerator(mixed_precision="fp16")  # use fp16 to reduce GPU memory

# Load tokenizer from the checkpoint directory to ensure vocabulary consistency
tokenizer = AutoTokenizer.from_pretrained(config['model_str'])
# Verify tokenizer loaded correctly
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
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
print(f"Model vocabulary size before resize: {model.config.vocab_size}")
print(f"Model embedding size before resize: {model.model.shared.num_embeddings}")

# Only resize if there's a mismatch
if len(tokenizer) != model.model.shared.num_embeddings:
    print(f"Resizing model embeddings from {model.model.shared.num_embeddings} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
else:
    print("Model and tokenizer vocabulary sizes match - no resize needed")

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

print(f"\n{'='*60}")
print(f"YAKE + BART Training - {config['num_epochs']} Epochs")
print(f"{'='*60}")
print(f"Model: {config['model_str']}")
print(f"Training data: {config['data_files']['train']}")
print(f"Epochs: {config['num_epochs']}")
print(f"Batch size: {config['batch_size']}")
print(f"Learning rate: {config['lr']}")
print(f"Total training steps: {num_training_steps}")
print(f"{'='*60}\n")

print("Training...")
best_rougeL = float('-inf')
best_epoch = -1
metrics_log = []

for epoch in range(config['num_epochs']):
    print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
    print("-" * 60)
    
    model.train()
    train_loss = 0
    train_steps = 0
    
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
                train_loss += loss.item()
                train_steps += 1

    avg_train_loss = train_loss / max(train_steps, 1)
    print(f"Average training loss: {avg_train_loss:.4f}")

    # evaluation
    print("Evaluating on validation set...")
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
    
    epoch_metrics = {
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        **result
    }
    metrics_log.append(epoch_metrics)
    
    print(f"Validation metrics: {result}")
    
    # Save every epoch
    #accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    #epoch_dir = os.path.join(config['output_dir'], f"epoch_{epoch + 1}")
    #unwrapped_model.save_pretrained(epoch_dir, save_function=accelerator.save)
    #tokenizer.save_pretrained(epoch_dir)
    #print(f"✓ Model saved to {epoch_dir}")
    
    # Track and save best model based on ROUGE-L
    if result.get('rougeL', 0) > best_rougeL:
        best_rougeL = result.get('rougeL', 0)
        best_epoch = epoch + 1
        best_epoch_dir = os.path.join(config['output_dir'], "Best_epoch")
        unwrapped_model.save_pretrained(best_epoch_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(best_epoch_dir)
        print(f"✓ Best model updated! ROUGE-L: {best_rougeL:.4f}")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best epoch: {best_epoch}")
print(f"Best ROUGE-L: {best_rougeL:.4f}\n")

# Print training summary
print("Training Summary:")
print("-" * 60)
for metrics in metrics_log:
    epoch_num = metrics['epoch']
    train_loss = metrics['train_loss']
    rouge1 = metrics.get('rouge1', 0)
    rouge2 = metrics.get('rouge2', 0)
    rougeL = metrics.get('rougeL', 0)
    print(f"Epoch {epoch_num}: Train Loss={train_loss:.4f}, "
          f"ROUGE-1={rouge1:.2f}, ROUGE-2={rouge2:.2f}, ROUGE-L={rougeL:.2f}")

# Save metrics to JSON
metrics_path = os.path.join(config['output_dir'], 'training_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_log, f, indent=2)
print(f"\n✓ Metrics saved to {metrics_path}")

# Save config to output dir
update_config(config['output_dir'], config)
print(f"✓ Config saved to {os.path.join(config['output_dir'], 'config.json')}")
