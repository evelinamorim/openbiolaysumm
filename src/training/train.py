import time
import torch, sys
from torch.optim import AdamW
import evaluate
from accelerate import Accelerator
import numpy as np
from model import LEDKForConditionalGeneration, GraphEncoder
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler
from src.utils.utils import get_processed_elife_data, load_train_config, update_config
import json
import os
os.environ['NLTK_DATA'] = 'nltk_data'  # or another directory you have write access to
import nltk
nltk.download('punkt_tab', force=True)
print(nltk.data.path)

print("Starting train.py...")

config_path = sys.argv[1]

# Config
config = load_train_config(config_path)

# load tools
ds = "elife"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: "%s"' % device)
accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")#config['model_str'])
metric = evaluate.load("rouge")

train_dataloader = get_processed_elife_data(ds, tokenizer, config, "train", shuffle=True)
val_dataloader = get_processed_elife_data(ds, tokenizer, config, "val", shuffle=False)

print("Train split size:", len(train_dataloader.dataset))
print("Val split size:", len(val_dataloader.dataset))

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

print("Loading model...")

model = LEDKForConditionalGeneration.from_pretrained(
    config['model_str'], 
    use_cache=False, 
    is_merge_encoders=config['is_merge_encoders'], 
    is_graph_decoder=config['is_graph_decoder'],
)
model.resize_token_embeddings(len(tokenizer))
graph_encoder = GraphEncoder(config)

model.config.num_beams = config['num_beams']
model.config.max_length = config['decoder_max_length']
model.config.min_length = config['min_length']
model.config.length_penalty = config['length_penalty']
model.config.no_repeat_ngram_size = config['no_repeat_ngram_size']

optimizer = AdamW(model.parameters(), lr=config['lr'])

val_dataloader, model, optimizer = accelerator.prepare(
    val_dataloader, model, optimizer
)

epoch_measures = []

num_epochs = config.get("num_epochs", 3)  # Default to 3 if not set

best_rougeL = -1
output_dir = config.get("output_dir", ".")
os.makedirs(output_dir, exist_ok=True)

# Create subfolders for organization
desc_dir = os.path.join(output_dir, "descriptions")
val_dir = os.path.join(output_dir, "val_results")
os.makedirs(desc_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

total_start_time = time.time()
epoch_times = []
val_times = []

output_dir = config["output_dir"]
val_results_dir = os.path.join(output_dir, "val_results")
best_score_path = os.path.join(output_dir, "best_score.txt")

# --- Find current best ROUGE-L and epoch at startup ---
def find_best_rougeL(results_root):
    best_rougeL = -1
    best_file = None
    if not os.path.exists(results_root):
        return best_rougeL, best_file
    for fname in os.listdir(results_root):
        if fname.startswith("val_results_epoch") and fname.endswith(".json"):
            fpath = os.path.join(results_root, fname)
            with open(fpath) as f:
                data = json.load(f)
                rougeL = data.get("rougeL", -1)
                if rougeL > best_rougeL:
                    best_rougeL = rougeL
                    best_file = fname
    return best_rougeL, best_file

# --- Initialize best score ---
if os.path.exists(best_score_path):
    with open(best_score_path, "r") as f:
        line = f.read().strip()
        if "," in line:
            best_rougeL, best_epoch_file = line.split(",")
            best_rougeL = float(best_rougeL)
        else:
            best_rougeL = float(line)
            best_epoch_file = None
else:
    best_rougeL, best_epoch_file = find_best_rougeL(val_results_dir)
    # Save this info for future runs
    with open(best_score_path, "w") as f:
        if best_epoch_file:
            f.write(f"{best_rougeL},{best_epoch_file}")
        else:
            f.write(str(best_rougeL))

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    epoch_start_time = time.time()
    model.train()
    train_loss = 0
    train_steps = 0

    train_loop_start = time.time()
    for step, batch in enumerate(train_dataloader):
        step_start_time = time.time()  # Start timer

        aids = batch["idx"]
        graph_enc_out = []
        valid_indices = []
        for i, aid in enumerate(aids):
            graph_out = graph_encoder.forward(aid, "train", device)
            if graph_out is None:
                continue
            graph_enc_out.append(graph_out)
            valid_indices.append(i)
        if not graph_enc_out:
            continue
        # Stack pooled graph embeddings for the batch: [batch_size, GAT_embedding_size]
        graph_enc_out = torch.stack(graph_enc_out, dim=0).to(torch.float16)
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].shape[0] == len(aids):
                batch[key] = batch[key][valid_indices]
        del batch['idx']

        # Move tensors to device
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["labels"] = batch["labels"].to(device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            graph_encoder_outputs=graph_enc_out,
            labels=batch["labels"]
        )
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_steps += 1

        if step % 10 == 0:
            step_time = time.time() - step_start_time  # End timer
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} | Step time: {step_time:.2f} sec")
            # Print input and predicted token IDs for the first sample in the batch
            with torch.no_grad():
                pred_ids = torch.argmax(outputs.logits, dim=-1)

    train_loop_end = time.time()
    train_minutes = (train_loop_end - train_loop_start) / 60

    avg_train_loss = train_loss / max(1, train_steps)
    print(f"Epoch {epoch+1} finished. Average training loss: {avg_train_loss:.4f}")
    print(f"Training time for epoch {epoch+1}: {train_minutes:.2f} minutes")

    # --- Validation after each epoch ---
    val_start_time = time.time()
    model.eval()
    metric = evaluate.load("rouge")
    all_preds = []
    all_refs = []
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            aids = batch["idx"]
            graph_enc_out = []
            valid_indices = []
            for i, aid in enumerate(aids):
                graph_out = graph_encoder.forward(aid, "val", device)
                if graph_out is None:
                    continue
                graph_enc_out.append(graph_out)
                valid_indices.append(i)
            if not graph_enc_out:
                continue
            # Stack pooled graph embeddings for the batch: [batch_size, GAT_embedding_size]
            graph_enc_out = torch.stack(graph_enc_out, dim=0).to(torch.float16)
            for key in batch:
                if isinstance(batch[key], torch.Tensor) and batch[key].shape[0] == len(aids):
                    batch[key] = batch[key][valid_indices]
            del batch['idx']

            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["labels"] = batch["labels"].to(device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                graph_encoder_outputs=graph_enc_out,
                labels=batch["labels"]
            )
            generated_tokens = outputs.logits.argmax(-1)
            labels = batch["labels"]

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = np.where(generated_tokens != -100, generated_tokens, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            all_preds.extend(decoded_preds)
            all_refs.extend(decoded_labels)
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
    val_end_time = time.time()
    val_minutes = (val_end_time - val_start_time) / 60
    val_times.append(val_minutes)

    val_result = metric.compute(use_stemmer=True)
    val_result = {k: round(v * 100, 4) for k, v in val_result.items()}
    print(f"Validation results after epoch {epoch+1}: {val_result}")
    print(f"Validation time for epoch {epoch+1}: {val_minutes:.2f} minutes")

    # Save best model based on ROUGE-L
    current_rougeL = val_result.get("rougeL", 0)
    epoch_file = f"val_results_epoch{epoch+1}.json"

    if current_rougeL > best_rougeL:
        best_rougeL = current_rougeL
        best_epoch_file = epoch_file
        model.save_pretrained(os.path.join(output_dir, "best_model"))
        with open(best_score_path, "w") as f:
            f.write(f"{best_rougeL},{best_epoch_file}")
        print(f"Best model saved to {os.path.join(output_dir, 'best_model')} (ROUGE-L: {best_rougeL})")

    # Save validation results for this epoch
    val_save_path = os.path.join(val_dir, f"val_results_epoch{epoch+1}.json")
    with open(val_save_path, "w") as f:
        json.dump(val_result, f, indent=4)
    print(f"Saved validation results for epoch {epoch+1} to {val_save_path}")

    # Save generated descriptions for this epoch (all predictions and references)
    desc_save_path = os.path.join(desc_dir, f"descriptions_epoch{epoch+1}.json")
    with open(desc_save_path, "w") as f:
        json.dump({
            "predictions": all_preds,
            "references": all_refs
        }, f, indent=4)
    print(f"Saved generated descriptions for epoch {epoch+1} to {desc_save_path}")

    epoch_times.append((train_minutes, val_minutes, avg_train_loss))

# Save final model
model.save_pretrained(os.path.join(output_dir, "final_model"))
print(f"Final model saved to {os.path.join(output_dir, 'final_model')}")

total_end_time = time.time()
total_minutes = (total_end_time - total_start_time) / 60

print("\n--- Timing Summary ---")
for i, (train_m, val_m, avg_loss) in enumerate(epoch_times):
    print(f"Epoch {i+1}: Training: {train_m:.2f} min | Validation: {val_m:.2f} min | Avg loss: {avg_loss:.4f}")
print(f"Total script time: {total_minutes:.2f} minutes")