"""
Simplified script to visualize specific results from training output files.
Easy to customize for specific comparisons.
"""

import os
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")


def parse_validation_results(file_path):
    """Parse validation results from .out file."""
    results = {}
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r"Validation results after epoch (\d+): ({.*})", line)
            if match:
                epoch = int(match.group(1))
                metrics_str = match.group(2).replace("'", '"')
                try:
                    metrics = json.loads(metrics_str)
                    results[epoch] = metrics
                except:
                    pass
    return results


def quick_comparison(file_paths, labels, output_path='comparison.png'):
    """
    Quick comparison of multiple runs.
    
    Args:
        file_paths: List of paths to .out files
        labels: List of labels for each file
        output_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for file_path, label in zip(file_paths, labels):
            results = parse_validation_results(file_path)
            if results:
                epochs = sorted(results.keys())
                values = [results[e].get(metric, 0) for e in epochs]
                ax.plot(epochs, values, marker='o', linewidth=2, label=label, alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=11)
        ax.set_title(f'{metric.upper()} Evolution', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.show()


def final_scores_table(file_paths, labels, output_csv='final_scores.csv'):
    """
    Create a table with final epoch scores.
    
    Args:
        file_paths: List of paths to .out files
        labels: List of labels for each file
        output_csv: Where to save the CSV
    """
    data = []
    for file_path, label in zip(file_paths, labels):
        results = parse_validation_results(file_path)
        if results:
            last_epoch = max(results.keys())
            metrics = results[last_epoch]
            data.append({
                'Model': label,
                'Epochs': last_epoch,
                'ROUGE-1': metrics.get('rouge1', 0),
                'ROUGE-2': metrics.get('rouge2', 0),
                'ROUGE-L': metrics.get('rougeL', 0),
                'ROUGE-Lsum': metrics.get('rougeLsum', 0)
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values('ROUGE-1', ascending=False)
    df.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"\nFinal Scores:\n{df.to_string(index=False)}")
    print(f"\nSaved to {output_csv}")
    return df


def main():
    """
    Example usage - customize this section for your specific needs.
    """
    base_folder = r'' #path to results here
    
    # Example 1: Compare specific runs
    print("Example 1: Comparing specific runs")
    file_paths = [
        os.path.join(base_folder, '506330.out'),
        os.path.join(base_folder, '506401.out'),
        os.path.join(base_folder, '506576.out'),
    ]
    labels = ['Run 506330', 'Run 506401', 'Run 506576']
    
    # Only use files that exist
    existing_files = [(f, l) for f, l in zip(file_paths, labels) if os.path.exists(f)]
    if existing_files:
        file_paths, labels = zip(*existing_files)
        quick_comparison(file_paths, labels, 'thesis_comparison_3runs.png')
        final_scores_table(file_paths, labels, 'thesis_final_scores.csv')
    
    # Example 2: Compare all runs in folder
    print("\n\nExample 2: All runs in folder")
    all_files = list(Path(base_folder).glob('*.out'))[:5]  # First 5 files
    if all_files:
        file_paths = [str(f) for f in all_files]
        labels = [f.stem for f in all_files]
        quick_comparison(file_paths, labels, 'thesis_all_runs_comparison.png')
        final_scores_table(file_paths, labels, 'thesis_all_runs_scores.csv')


if __name__ == "__main__":
    main()
