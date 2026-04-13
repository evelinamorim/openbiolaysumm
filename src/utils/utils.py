import os, json
from torch.utils.data.dataloader import DataLoader
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader


def load_train_config(config_path):
    """
    Load config for training.
    """
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config

def load_dataset(config, split):
    json_file = config['data_files'][split]
    
    with open(json_file, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # JSON array format 
            data = json.load(f)
        else:
            # JSONL format 
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    return data


def load_dataset_abstract(ds, split):
    if split == "train":
        fp = "train_filtered.json"
    else:
        fp = f"{split}.json"
    with open(fp, "r") as f:
        data = json.loads(f.read())

    data = [dict(id=inst['id'],
            article=" ".join(inst['abstract']), 
            summary=" ".join(inst['summary'])) for inst in data]
    return data


def add_graph_text_data(graph_text_path, split, data):
    """
    Load augmented text data for the given dataset and split.
    """
    fp = f"{graph_text_path}/{split}_abstract_concepts_explanation.jsonl"
    with open(fp, "r") as f:
        explain_data = f.readlines()
        explain_data = [json.loads(line) for line in explain_data]

    for i, inst in enumerate(data):
        aid = inst['id']
        graph_explainations = explain_data[i]
        assert aid == graph_explainations['id']
        data[i]['article'] = "[GRAPH_FACTS]\n" + graph_explainations['text'] + "\n[ARTICLE]\n" + data[i]['article'] 

    return data
    
def update_config(save_dir, kg_config):
    config_path = f"{save_dir}/config.json"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(json.dumps(kg_config, indent=2))
    else:
        with open(config_path, "r") as f:
            config = json.loads(f.read())
        config.update(kg_config)
        with open(config_path, "w") as f:
            f.write(json.dumps(config, indent=2))
 
def get_processed_elife_data(ds, tokenizer, config, split, shuffle=False):
    
    def process_data_to_model_inputs(batch):
        """
        Process batch for BART training.
        Handles multiple data formats:
        - train_filtered.json: has 'abstract' as list
        - train_yake_bart.json: has 'article' as string (with keywords)
        """
        # Handle different field names
        if "abstract" in batch:
            abstracts = batch["abstract"]
        elif "article" in batch:
            abstracts = batch["article"]
        else:
            raise KeyError("Batch must have 'abstract' or 'article' field")
    
        summaries = batch["summary"]
    
        # Convert lists to strings if needed
        if abstracts and isinstance(abstracts[0], list):
            abstracts = [" ".join(sentences) for sentences in abstracts]
    
        if summaries and isinstance(summaries[0], list):
            summaries = [" ".join(sentences) for sentences in summaries]
    
        # Tokenize inputs
        inputs = tokenizer(
            abstracts,
            padding="max_length",
            truncation=True,
            max_length=config.get('encoder_max_length', 1024),
        )
    
        # Tokenize outputs
        outputs = tokenizer(
            summaries,
            padding="max_length",
            truncation=True,
            max_length=config.get('decoder_max_length', 256),
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # Replace padding with -100
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    data = load_dataset(config, split)

    data = [{"idx": i, **x} for i, x in enumerate(data)]

    if config['is_input_aug']:
        data = add_graph_text_data(config['graph_data_path'], "train", data)

    data = Dataset.from_list(data)

    # map train data
    data = data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=config['batch_size'],
    )

    # set Python list to PyTorch tensor
    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "idx"],
    )


    dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=shuffle)

    return dataloader