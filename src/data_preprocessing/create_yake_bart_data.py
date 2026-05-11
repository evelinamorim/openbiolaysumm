import json
import os
import argparse

"""
Script to augment elife training data with YAKE keywords filtered by DBpedia and Ollama.
Prepends biomedical keyword names to article text for BART training.
"""

def create_keyword_text(keywords, max_keywords=10):
    """Format top keywords as a bullet list."""
    formatted = [f"• {kw}" for kw in list(keywords)[:max_keywords] if kw]
    return "\n".join(formatted) if formatted else ""


def load_biomedical_keywords(biomedical_path):
    """Load biomedical keywords from ollama_filter.py output.
    Returns a dict mapping article_id -> list of keyword strings.
    """
    with open(biomedical_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    id_to_keywords = {}
    for entry in data:
        article_id = entry.get("id", "")
        # biomedical_keywords is a dict {keyword: {link, description}}
        keywords = list(entry.get("biomedical_keywords", {}).keys())
        if article_id:
            id_to_keywords[article_id] = keywords

    return id_to_keywords


def load_input_data(input_path):
    """Load input JSON or JSONL file."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {line[:50]}... Error: {e}")
    return data


def augment_data(input_path, biomedical_path, output_path):
    print(f"Loading biomedical keywords from: {biomedical_path}")
    id_to_keywords = load_biomedical_keywords(biomedical_path)
    print(f"Loaded keywords for {len(id_to_keywords)} articles.")

    print(f"Loading input data from: {input_path}")
    original_data = load_input_data(input_path)
    print(f"Loaded {len(original_data)} articles.")

    augmented_data = []
    no_keywords_count = 0

    for item in original_data:
        article_id = item.get("id", "")

        # Build article text
        if "article" in item:
            base_article = item["article"]
        else:
            abstract_text = " ".join(item.get("abstract", [])) if isinstance(item.get("abstract"), list) else item.get("abstract", "")
            sections = item.get("sections", [])
            sections_text = "\n".join([" ".join(s) if isinstance(s, list) else str(s) for s in sections]) if sections else ""
            base_article = f"{abstract_text}\n{sections_text}".strip()

        # Prepend biomedical keywords if available
        keywords = id_to_keywords.get(article_id, [])
        if keywords:
            keyword_text = create_keyword_text(keywords)
            enhanced_article = f"Key concepts and definitions:\n{keyword_text}\n\n{base_article}"
        else:
            enhanced_article = base_article
            no_keywords_count += 1

        # Get summary
        summary_text = item.get("summary", "")
        if isinstance(summary_text, list):
            summary_text = " ".join(summary_text)

        augmented_data.append({
            "id": article_id,
            "article": enhanced_article,
            "summary": summary_text,
        })

    # Write output as JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✓ Augmentation complete.")
    print(f"  Total articles    : {len(augmented_data)}")
    print(f"  With keywords     : {len(augmented_data) - no_keywords_count}")
    print(f"  Without keywords  : {no_keywords_count}")
    print(f"  Output written to : {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment eLife data with YAKE biomedical keywords for BART training."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input data file (train_filtered.json).",
    )
    parser.add_argument(
        "--biomedical",
        required=True,
        help="Path to biomedical keywords file produced by ollama_filter.py (*_biomedical.json).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file (e.g. data/train_yake_bart.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        exit(1)
    if not os.path.isfile(args.biomedical):
        print(f"Error: biomedical file not found: {args.biomedical}")
        exit(1)

    augment_data(args.input, args.biomedical, args.output)