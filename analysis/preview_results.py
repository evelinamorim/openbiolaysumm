"""
Preview script to see what data will be extracted from .out files.
No visualization libraries required - just shows the data.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict


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


def parse_training_losses(file_path):
    """Parse training losses from .out file."""
    losses = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r"Epoch (\d+) \| Step (\d+) \| Loss: ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = float(match.group(3))
                losses.append((epoch, step, loss))
    return losses


def preview_folder(folder_path):
    """Preview all .out files in a folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return
    
    out_files = list(folder.glob("*.out"))
    
    if not out_files:
        print(f"No .out files found in {folder_path}")
        return
    
    print(f"\nFound {len(out_files)} .out files")
    print("=" * 80)
    
    summary_data = []
    
    for out_file in sorted(out_files):
        validation_results = parse_validation_results(out_file)
        training_losses = parse_training_losses(out_file)
        
        if validation_results or training_losses:
            print(f"\n📄 {out_file.name}")
            print("-" * 80)
            
            # Validation results
            if validation_results:
                print("\n  Validation Results:")
                for epoch in sorted(validation_results.keys()):
                    metrics = validation_results[epoch]
                    print(f"    Epoch {epoch}:")
                    print(f"      ROUGE-1:    {metrics.get('rouge1', 0):6.4f}")
                    print(f"      ROUGE-2:    {metrics.get('rouge2', 0):6.4f}")
                    print(f"      ROUGE-L:    {metrics.get('rougeL', 0):6.4f}")
                    print(f"      ROUGE-Lsum: {metrics.get('rougeLsum', 0):6.4f}")
                
                # Store summary
                last_epoch = max(validation_results.keys())
                summary_data.append({
                    'file': out_file.name,
                    'epochs': last_epoch,
                    'rouge1': validation_results[last_epoch].get('rouge1', 0),
                    'rouge2': validation_results[last_epoch].get('rouge2', 0),
                    'rougeL': validation_results[last_epoch].get('rougeL', 0),
                    'rougeLsum': validation_results[last_epoch].get('rougeLsum', 0),
                })
            else:
                print("  ⚠ No validation results found")
            
            # Training losses
            if training_losses:
                print(f"\n  Training Losses: {len(training_losses)} steps recorded")
                print(f"    First loss: {training_losses[0][2]:.4f}")
                print(f"    Last loss:  {training_losses[-1][2]:.4f}")
            else:
                print("  ⚠ No training losses found")
    
    # Summary table
    if summary_data:
        print("\n\n" + "=" * 80)
        print("SUMMARY TABLE - Final Epoch Results")
        print("=" * 80)
        print(f"\n{'File':<20} {'Epochs':>7} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10} {'ROUGE-Lsum':>10}")
        print("-" * 80)
        
        # Sort by ROUGE-1
        summary_data.sort(key=lambda x: x['rouge1'], reverse=True)
        
        for row in summary_data:
            print(f"{row['file']:<20} {row['epochs']:>7} {row['rouge1']:>10.4f} {row['rouge2']:>10.4f} "
                  f"{row['rougeL']:>10.4f} {row['rougeLsum']:>10.4f}")
        
        # Best scores
        print("\n" + "=" * 80)
        print("BEST SCORES")
        print("=" * 80)
        best_rouge1 = max(summary_data, key=lambda x: x['rouge1'])
        best_rouge2 = max(summary_data, key=lambda x: x['rouge2'])
        best_rougeL = max(summary_data, key=lambda x: x['rougeL'])
        
        print(f"Best ROUGE-1:    {best_rouge1['rouge1']:.4f} ({best_rouge1['file']})")
        print(f"Best ROUGE-2:    {best_rouge2['rouge2']:.4f} ({best_rouge2['file']})")
        print(f"Best ROUGE-L:    {best_rougeL['rougeL']:.4f} ({best_rougeL['file']})")
        
        # Average scores
        avg_rouge1 = sum(x['rouge1'] for x in summary_data) / len(summary_data)
        avg_rouge2 = sum(x['rouge2'] for x in summary_data) / len(summary_data)
        avg_rougeL = sum(x['rougeL'] for x in summary_data) / len(summary_data)
        
        print(f"\nAverage ROUGE-1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-2: {avg_rouge2:.4f}")
        print(f"Average ROUGE-L: {avg_rougeL:.4f}")


def main():
    """Main preview function."""
    base_folder = r'' #path to results here
    
    print("=" * 80)
    print("DATA PREVIEW - What will be extracted from your .out files")
    print("=" * 80)
    print(f"\nScanning folder: {base_folder}")
    
    preview_folder(base_folder)
    
    print("\n\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("\n1. Install visualization packages:")
    print("   python setup_visualization.py")
    print("\n2. Generate visualizations:")
    print("   python visualize_results.py")
    print("   or")
    print("   python visualize_results_simple.py")
    print("\n3. Check VISUALIZATION_README.md for detailed instructions")


if __name__ == "__main__":
    main()
