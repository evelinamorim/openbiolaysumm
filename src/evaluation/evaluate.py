import torch
from transformers import AutoTokenizer
from src.models.model import LEDKForConditionalGeneration, GraphEncoder
from utils import get_processed_elife_data, load_train_config
import evaluate
import json
import os
import numpy as np

# Load config and tokenizer
config = load_train_config("../../configs/train_config.json")
tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load best model
model = LEDKForConditionalGeneration.from_pretrained(
    os.path.join(config['output_dir'], "best_model"),
    use_cache=False,
    is_merge_encoders=config['is_merge_encoders'],
    is_graph_decoder=config['is_graph_decoder'],
)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()

graph_encoder = GraphEncoder(config)

# Load test dataloader
test_dataloader = get_processed_elife_data("elife", tokenizer, config, "test", shuffle=False)

metric = evaluate.load("rouge")
all_preds = []
all_refs = []

with torch.no_grad():
    for batch in test_dataloader:
        aids = batch["idx"]
        graph_enc_out = []
        valid_indices = []
        for i, aid in enumerate(aids):
            graph_out = graph_encoder.forward(aid, "test", device)
            if graph_out is None:
                continue
            graph_enc_out.append(graph_out)
            valid_indices.append(i)
        if not graph_enc_out:
            continue
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

        generated_tokens = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = np.where(generated_tokens != -100, generated_tokens, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Use your postprocess_text function if needed
        # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        all_preds.extend(decoded_preds)
        all_refs.extend(decoded_labels)
        metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels,
        )

test_result = metric.compute(use_stemmer=True)
test_result = {k: round(v * 100, 4) for k, v in test_result.items()}
print(f"Test results: {test_result}")

# Optionally, save predictions and references
with open(os.path.join(config['output_dir'], "test_descriptions_new.json"), "w") as f:
    json.dump({"predictions": all_preds, "references": all_refs}, f, indent=4)
with open(os.path.join(config['output_dir'], "test_results_new.json"), "w") as f:
    json.dump(test_result, f, indent=4)
