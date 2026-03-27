import json
import statistics
from pathlib import Path

def word_count(text: str) -> int:
    return len(text.split())

def analyze_jsonl(path: Path):
    counts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            counts.append(word_count(text))
    n = len(counts)
    total_words = sum(counts)
    mean = statistics.mean(counts) if n else 0
    stdev = statistics.stdev(counts) if n > 1 else 0
    return n, total_words, mean, stdev, counts

def main():
    files = [
        Path("Dset_downsample/test.jsonl"),
        Path("Dset_downsample/train.jsonl"),
    ]
    for p in files:
        if not p.exists():
            print(f"Missing: {p}")
            continue
        n, total_words, mean, stdev, counts = analyze_jsonl(p)
        print(f"\n{p.name}")
        print(f"  Total items: {n}")
        print(f"  Total words: {total_words}")
        print(f"  Mean word count: {mean:.2f}")
        print(f"  Std deviation: {stdev:.2f}")
        print(f"  Min: {min(counts) if counts else 0}, Max: {max(counts) if counts else 0}")

if __name__ == "__main__":
    main()