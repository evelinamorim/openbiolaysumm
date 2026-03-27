import pandas as pd

train = pd.read_csv("elife_raw/train.csv")
test = pd.read_csv("elife_raw/test.csv")

# new: dataset size / overlap summary
n_train = len(train)
n_test = len(test)
total = n_train + n_test
print(f"Train entries: {n_train}")
print(f"Test entries: {n_test}")
print(f"Total entries (train+test): {total}")

# unique texts across both splits
if "text" in train.columns and "text" in test.columns:
    unique_texts = pd.concat([train["text"], test["text"]]).nunique()
    print(f"Unique texts in full dataset: {unique_texts}")

    # overlap between train and test (exact same text)
    overlap_texts = set(train["text"]).intersection(set(test["text"]))
    print(f"Overlap (same text) between train and test: {len(overlap_texts)}")

    # duplicates inside combined
    combined = pd.concat([train, test], ignore_index=True)
    dup_within = combined["text"].duplicated().sum()
    print(f"Duplicate texts within combined dataset: {dup_within}")
else:
    print("No 'text' column found to compute overlaps/uniques.")

print("Train label distribution:\n", train["label"].value_counts(normalize=True))
print("Test label distribution:\n", test["label"].value_counts(normalize=True))
train["len"] = train["text"].str.len()
test["len"] = test["text"].str.len()
print(train["len"].describe())
print(test["len"].describe())
from collections import Counter

def top_words(texts, n=20):
    words = " ".join(texts).split()
    return Counter(words).most_common(n)

print("Top words in label 0:", top_words(train[train["label"]==0]["text"]))
print("Top words in label 1:", top_words(train[train["label"]==1]["text"]))