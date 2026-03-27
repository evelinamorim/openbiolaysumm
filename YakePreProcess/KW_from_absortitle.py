import json
import os

# Define file paths
processed_files = {
    "val": "File_pre_processed/processed_val_keywords.json",
    "test": "File_pre_processed/processed_test_keywords.json",
    "train": "Files_pre_processed/processed_train_keywords.json"
}

original_files = {
    "val": "../val.json",  # Adjusted to point outside the YakePreProcess directory
    "test": "../test.json",
    "train": "../train.json"
}

def check_keyword_source(processed_file, original_file):
    # Load processed keywords
    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    # Load original articles
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    results = []

    # Iterate through processed keywords
    for entry in processed_data:
        article_id = entry["id"]
        keywords = entry["keywords"]

        # Find the corresponding article in the original data
        original_article = next((article for article in original_data if article["id"] == article_id), None)
        if not original_article:
            continue

        # Get the title and abstract
        title = original_article.get("title", "").lower()

        # Handle the abstract field if it's a list
        abstract = original_article.get("abstract", "")
        if isinstance(abstract, list):
            abstract = " ".join(abstract)  # Join list elements into a single string
        abstract = abstract.lower()

        # Handle the sections field if it exists
        sections = original_article.get("sections", [])
        if isinstance(sections, list):
            # Flatten the sections list and ensure all elements are strings
            flattened_sections = []
            for section in sections:
                if isinstance(section, list):
                    flattened_sections.extend(section)  # Add elements of the inner list
                else:
                    flattened_sections.append(section)  # Add the string directly

            # Join all section texts into a single string
            sections_text = " ".join(map(str, flattened_sections)).lower()
        else:
            sections_text = ""

        for keyword in keywords:
            keyword_lower = keyword.lower()
            in_title = keyword_lower in title
            in_abstract = keyword_lower in abstract
            in_sections = keyword_lower in sections_text

            if in_title and in_abstract and in_sections:
                source = "all"  # Found in title, abstract, and sections
            elif in_title and in_abstract:
                source = "title and abstract"
            elif in_title and in_sections:
                source = "title and sections"
            elif in_abstract and in_sections:
                source = "abstract and sections"
            elif in_title:
                source = "title"
            elif in_abstract:
                source = "abstract"
            elif in_sections:
                source = "sections"
            else:
                source = "none"

            results.append({
                "id": article_id,
                "keyword": keyword,
                "source": source
            })
    
    return results

# Prompt user to choose which file to process
print("Choose which dataset to process:")
print("1. val")
print("2. test")
print("3. train")
choice = input("Enter the number corresponding to your choice: ")

dataset_map = {
    "1": "val",
    "2": "test",
    "3": "train"
}

selected_dataset = dataset_map.get(choice)
if not selected_dataset:
    print("Invalid choice. Exiting.")
    exit()

# Process the selected dataset
processed_file = processed_files[selected_dataset]
original_file = original_files[selected_dataset]

print(f"Processing {selected_dataset} dataset...")
results = check_keyword_source(processed_file, original_file)

# Ensure output directory exists inside YakePreProcess
output_dir = "KeywordSources"
os.makedirs(output_dir, exist_ok=True)

# Save results to a new file
output_file = f"{output_dir}/keyword_sources_{selected_dataset}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {output_file}")