import json
import os
from pathlib import Path

"""
Script to augment elife training data with YAKE keywords and DBPedia descriptions.
This combines article text with YAKE-extracted keywords and their DBPedia descriptions
to create augmented input for BART training.
"""

# Path to the combined YAKE + DBPedia keywords
YAKE_DATA_PATH = "../../YakePreProcess/combined/combined_keywords_grouped.json"

# Input/Output paths
INPUT_DATA = {
    "train": "train_filtered.json",
    "val": "val.json",
    "test": "test.json"
}

OUTPUT_DATA = {
    "train": "train_yake_bart.json",
    "val": "val_yake_bart.json",
    "test": "test_yake_bart.json"
}

def load_yake_keywords():
    """Load YAKE keywords and their descriptions from combined file."""
    with open(YAKE_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_keyword_description_text(keywords_list, max_keywords=10):
    """
    Create a text representation of top keywords from YAKE.
    
    Args:
        keywords_list: List of keyword dicts with 'keyword' field
        max_keywords: Maximum number of keywords to include
    
    Returns:
        String with formatted keywords only
    """
    if not keywords_list:
        return ""
    
    # Sort by position (first keywords are more important from YAKE)
    formatted_keywords = []
    for i, kw_dict in enumerate(keywords_list[:max_keywords]):
        keyword = kw_dict.get("keyword", "")
        
        # Only include if we have meaningful content
        if keyword:
            formatted_keywords.append(f"• {keyword}")
    
    if formatted_keywords:
        return "\n".join(formatted_keywords)
    return ""

def augment_data_with_yake(split, yake_data, input_path, output_path):
    """
    Augment training/val/test data with YAKE keywords and descriptions.
    
    Args:
        split: "train", "val", or "test"
        yake_data: Dictionary containing YAKE keywords grouped by split
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    """
    # Load original data
    original_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        # Try loading as JSON array first
        f.seek(0)
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # It's a JSON array
            original_data = json.load(f)
        else:
            # It's JSONL (one JSON object per line)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        original_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {line[:50]}... Error: {e}")
                        continue
    
    # Get YAKE keywords for this split
    yake_keywords = yake_data.get(split, [])
    
    # Create a mapping from article ID to keywords
    id_to_keywords = {}
    for entry in yake_keywords:
        article_id = entry.get("id", "")
        keywords = entry.get("keywords", [])
        if article_id:
            id_to_keywords[article_id] = keywords
    
    # Augment each article with YAKE keywords
    augmented_data = []
    for item in original_data:
        article_id = item.get("id", "")
        
        # Get keywords for this article if available
        keywords_for_article = id_to_keywords.get(article_id, [])
        
        # Build article text from abstract + sections (matching utils.py load_dataset format)
        if "article" in item:
            # Already has article field
            base_article = item["article"]
        else:
            # Build from abstract and sections like utils.py does
            abstract_text = " ".join(item.get("abstract", [])) if isinstance(item.get("abstract"), list) else item.get("abstract", "")
            sections = item.get("sections", [])
            if isinstance(sections, list) and sections:
                # sections is a list of lists of sentences
                sections_text = "\n".join([" ".join(s) if isinstance(s, list) else str(s) for s in sections])
            else:
                sections_text = ""
            
            base_article = f"{abstract_text}\n{sections_text}".strip()
        
        # Prepend YAKE keywords if available
        enhanced_article = base_article
        if keywords_for_article:
            keyword_text = create_keyword_description_text(keywords_for_article, max_keywords=10)
            if keyword_text:
                # Prepend keyword information to the article
                enhanced_article = f"Key concepts and definitions:\n{keyword_text}\n\n{base_article}"
        
        # Get summary
        summary_text = item.get("summary", "")
        if isinstance(summary_text, list):
            summary_text = " ".join(summary_text)
        
        # Create augmented item
        augmented_item = {
            "id": article_id,
            "article": enhanced_article,
            "summary": summary_text
        }
        
        augmented_data.append(augmented_item)
    
    # Write augmented data
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Augmented {split} split: {len(augmented_data)} samples written to {output_path}")
    
    return len(augmented_data)

def main():
    print("Loading YAKE keywords...")
    yake_data = load_yake_keywords()
    print(f"Loaded YAKE data for splits: {list(yake_data.keys())}")
    
    print("\nAugmenting data with YAKE keywords and DBPedia descriptions...\n")
    
    total_samples = {}
    for split in ["train", "val", "test"]:
        if INPUT_DATA[split] and os.path.exists(INPUT_DATA[split]):
            count = augment_data_with_yake(
                split,
                yake_data,
                INPUT_DATA[split],
                OUTPUT_DATA[split]
            )
            total_samples[split] = count
        else:
            print(f"Skipping {split} - file not found: {INPUT_DATA[split]}")
    
    print(f"\n✓ Data augmentation complete!")
    print(f"  - Train: {total_samples.get('train', 0)} samples")
    print(f"  - Val: {total_samples.get('val', 0)} samples")
    print(f"  - Test: {total_samples.get('test', 0)} samples")
    print(f"\nOutput files:")
    for split in ["train", "val", "test"]:
        if total_samples.get(split):
            print(f"  - {OUTPUT_DATA[split]}")

if __name__ == "__main__":
    main()
