import json

# Load the 50 selected articles from filtered_train_keywords.json
with open("filtered_train_keywords.json", "r") as file:
    selected_articles = json.load(file)

# Extract the list of IDs from the selected articles
selected_ids = {article["id"] for article in selected_articles}

# Load the processed_train_bigrams.json file from the correct folder
with open("../Files_pre_processed/processed_train_bigrams.json", "r", encoding="utf-8") as file:
    bigrams_data = json.load(file)

# Create a dictionary for fast lookup of bigram articles by ID
bigrams_dict = {article["id"]: article for article in bigrams_data}

# Extract the bigrams for the selected articles while maintaining the order from filtered_train_keywords
filtered_bigrams = [bigrams_dict[article["id"]] for article in selected_articles if article["id"] in bigrams_dict]

# Save the filtered bigrams into a new file
with open("filtered_train_bigrams.json", "w") as file:
    json.dump(filtered_bigrams, file, indent=4)

print("Filtered bigrams saved to filtered_train_bigrams.json")
