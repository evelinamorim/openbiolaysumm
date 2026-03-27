import os
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

STOPWORDS = {"Figure", "Figure supplement", "Supplementary Figure" }

# --- Load and Save ---
def load_keywords(split):
    path = f"YakePreProcess/filter_{split}/filtered_{split}_bigrams.json"
    with open(path) as f:
        return json.load(f)

def load_candidates(split):
    path = f"DBpedia_Candidates/{split}_dbpedia_candidates.json"
    with open(path) as f:
        return json.load(f)

def save_similarity_results(results, split):
    os.makedirs("similarities", exist_ok=True)
    out_path = f"similarities/DB_Descriptions_matches/{split}_scibert_matches_descriptions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# --- SciBERT Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
model.eval()

def embed_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# --- Main Similarity Logic ---
def compute_similarity(split):
    print(f"Running SciBERT similarity for split '{split}'...")
    keywords_data = load_keywords(split)  # Now this loads filtered_{split}_bigrams.json
    dbpedia_data = load_candidates(split)
    results = {}

    for doc in tqdm(keywords_data):
        doc_id = doc["id"]
        results[doc_id] = []

        for kw, info in doc.get("dbpedia", {}).items():
            kw_clean = kw.strip()
            if not kw_clean or kw_clean in STOPWORDS:
                continue

            kw_embedding = embed_text(kw_clean)
            candidates = dbpedia_data.get(kw_clean, [])

            if not candidates:
                continue  # skip if no candidates

            scored = []
            for cand in candidates:
                desc = cand.get("description", "")
                url = cand.get("url", "")
                try:
                    desc_embedding = embed_text(desc)
                    score = cosine_similarity([kw_embedding], [desc_embedding])[0][0]
                    scored.append({
                        "score": float(score),
                        "url": url,
                        "description": desc
                    })
                except Exception as e:
                    print(f"Skipped: Error embedding candidate: {url} | {e}")

            if scored:
                scored.sort(key=lambda x: x["score"], reverse=True)
                best_match = scored[0]
                worst_match = scored[-1]
                # Only keep if best score >= 0.5
                if best_match["score"] < 0.5:
                    continue
            else:
                continue  # skip if no valid scores

            results[doc_id].append({
                "keyword": kw_clean,
                "best_match": best_match,
                "worst_match": worst_match
            })

    save_similarity_results(results, split)
    print(f"Results saved to DBpedia_Matches/{split}_scibert_matches.json")

# --- Entry Point ---
if __name__ == "__main__":
    split_map = {"1": "val", "2": "test", "3": "train"}
    print("Choose the split to run similarity:\n1. val\n2. test\n3. train")
    split_choice = input("Your split number: ").strip()
    split = split_map.get(split_choice, "val")

    compute_similarity(split)
