"""
Script to visualize model training results from output files.
Generates line graphs, bar charts, and tables for thesis.
"""
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def parse_validation_results(file_path):
    """
    Parse validation results from a .out file.
    
    Returns:
        dict: {epoch: {metric: value}}
    """
    results = {}
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Look for validation results patterns
            match = re.search(r"Validation results after epoch (\d+): ({.*})", line)
            if not match:
                match = re.search(r"Epoch (\d+) metric result: ({.*})", line)
            if match:
                epoch = int(match.group(1))
                # Parse the dictionary string
                metrics_str = match.group(2)
                # Use ast.literal_eval or json.loads after replacing single quotes
                metrics_str = metrics_str.replace("'", '"')
                try:
                    metrics = json.loads(metrics_str)
                    results[epoch] = metrics
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse metrics in {file_path}: {metrics_str}")
    
    return results


def parse_training_losses(file_path):
    """
    Parse training losses from a .out file.
    
    Returns:
        list: [(epoch, step, loss, time)]
    """
    losses = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Look for loss pattern: "Epoch X | Step Y | Loss: Z.ZZZZ | Step time: W.WW sec"
            match = re.search(r"Epoch (\d+) \| Step (\d+) \| Loss: ([\d.]+) \| Step time: ([\d.]+) sec", line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = float(match.group(3))
                time = float(match.group(4))
                losses.append((epoch, step, loss, time))
    
    return losses


def collect_all_results(base_folder, recursive=False):
    """
    Collect all results from .out files in a folder.

    Args:
        recursive: bool - if True, search subfolders for .out files.

    Returns:
        dict: {run_key: {'validation': {...}, 'training': [...]}}
    """
    all_results = {}

    folder_path = Path(base_folder)
    out_files = folder_path.rglob("*.out") if recursive else folder_path.glob("*.out")
    for out_file in out_files:
        validation_results = parse_validation_results(out_file)
        training_losses = parse_training_losses(out_file)

        if validation_results or training_losses:
            if recursive:
                rel_key = out_file.relative_to(folder_path).as_posix()
                run_key = rel_key.replace("/", "__").replace(".out", "")
            else:
                run_key = out_file.stem

            all_results[run_key] = {
                'validation': validation_results,
                'training': training_losses,
                'file_path': str(out_file)
            }

    return all_results


def plot_training_curves(results_dict, output_folder, title_prefix=""):
    """
    Create line plots showing training loss over steps.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Plot individual runs
    for run_name, data in results_dict.items():
        if not data['training']:
            continue
            
        training_data = data['training']
        epochs, steps, losses, times = zip(*training_data)
        
        # Create global step count
        global_steps = list(range(len(steps)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss over steps
        ax1.plot(global_steps, losses, linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{title_prefix}Training Loss - {run_name}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Step time over steps
        ax2.plot(global_steps, times, linewidth=1.5, alpha=0.7, color='orange')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Time per Step (seconds)')
        ax2.set_title(f'{title_prefix}Step Time - {run_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_folder / f'training_curves_{run_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot all runs together
    fig, ax = plt.subplots(figsize=(12, 6))
    for run_name, data in results_dict.items():
        if not data['training']:
            continue
        training_data = data['training']
        _, steps, losses, _ = zip(*training_data)
        global_steps = list(range(len(steps)))
        ax.plot(global_steps, losses, linewidth=1, alpha=0.5, label=run_name)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{title_prefix}Training Loss - All Runs')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_folder / 'training_curves_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_folder}")


def plot_validation_evolution(results_dict, output_folder, title_prefix=""):
    """
    Create line plots showing validation metrics evolution across epochs.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect all metrics across all runs
    metrics_data = defaultdict(lambda: defaultdict(list))
    
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        for epoch, metrics in sorted(data['validation'].items()):
            for metric_name, value in metrics.items():
                metrics_data[metric_name][run_name].append((epoch, value))
    
    # Plot each metric
    for metric_name, runs_data in metrics_data.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for run_name, values in runs_data.items():
            epochs, scores = zip(*sorted(values))
            ax.plot(epochs, scores, marker='o', linewidth=2, label=run_name, alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{metric_name.upper()} Score')
        ax.set_title(f'{title_prefix}Validation {metric_name.upper()} Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(output_folder / f'validation_{metric_name}_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved validation evolution plots to {output_folder}")


def plot_final_comparison_bars(results_dict, output_folder, title_prefix="", only_epochs=None, output_suffix=""):
    """
    Create bar charts comparing final epoch results across runs.

    Args:
        only_epochs: int | None - if set, include only runs whose last epoch matches this value.
        output_suffix: str - suffix for output filenames (e.g., "_epoch20").
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Extract final epoch results for each run
    final_results = {}
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        # Get last epoch
        last_epoch = max(data['validation'].keys())
        if only_epochs is not None and last_epoch != only_epochs:
            continue
        final_results[run_name] = data['validation'][last_epoch]
    
    if not final_results:
        print("No validation results found for bar chart")
        return
    
    # Create DataFrame
    df = pd.DataFrame(final_results).T
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    df.plot(kind='bar', ax=ax, width=0.8, rot=45, alpha=0.8)
    ax.set_xlabel('Run')
    ax.set_ylabel('Score')
    ax.set_title(f'{title_prefix}Final Validation Results Comparison')
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_folder / f'final_comparison_bars{output_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot individual metrics
    for metric in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        df[metric].plot(kind='bar', ax=ax, color='steelblue', alpha=0.8, rot=45)
        ax.set_xlabel('Run')
        ax.set_ylabel(f'{metric.upper()} Score')
        ax.set_title(f'{title_prefix}Final {metric.upper()} Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_folder / f'final_{metric}_comparison{output_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved comparison bar charts to {output_folder}")


def create_results_table(results_dict, output_folder, title_prefix=""):
    """
    Create a formatted table with final results and save as CSV and image.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Extract final epoch results
    table_data = []
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        last_epoch = max(data['validation'].keys())
        metrics = data['validation'][last_epoch]
        
        row = {
            'Run': run_name,
            'Epochs': last_epoch,
            'ROUGE-1': metrics.get('rouge1', 0),
            'ROUGE-2': metrics.get('rouge2', 0),
            'ROUGE-L': metrics.get('rougeL', 0),
            'ROUGE-Lsum': metrics.get('rougeLsum', 0),
        }
        table_data.append(row)
    
    if not table_data:
        print("No data for table")
        return
    
    df = pd.DataFrame(table_data)
    
    # Sort by ROUGE-1 descending
    df = df.sort_values('ROUGE-1', ascending=False)
    
    # Save as CSV
    csv_path = output_folder / 'final_results_table.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved results table to {csv_path}")
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with color coding
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best scores in each column
    for col_idx, col_name in enumerate(df.columns):
        if col_name in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']:
            best_val = df[col_name].max()
            for row_idx, val in enumerate(df[col_name]):
                if val == best_val:
                    table[(row_idx + 1, col_idx)].set_facecolor('#90EE90')
                    table[(row_idx + 1, col_idx)].set_text_props(weight='bold')
    
    plt.title(f'{title_prefix}Final Validation Results', fontsize=14, weight='bold', pad=20)
    plt.savefig(output_folder / 'final_results_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved results table image to {output_folder}")
    
    return df


def create_best_results_table(results_dict, output_folder, title_prefix="", metric_for_best='rougeL'):
    """
    Create a formatted table with BEST epoch results (based on highest metric) and save as CSV and image.
    
    Args:
        metric_for_best: str - metric to use for determining best epoch (default: 'rougeL')
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Extract best epoch results
    table_data = []
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        # Find epoch with best metric score
        best_epoch = None
        best_score = -1
        for epoch, metrics in data['validation'].items():
            score = metrics.get(metric_for_best, 0)
            if score > best_score:
                best_score = score
                best_epoch = epoch
        
        if best_epoch is None:
            continue
            
        metrics = data['validation'][best_epoch]
        
        row = {
            'Run': run_name,
            'Best Epoch': best_epoch,
            'ROUGE-1': metrics.get('rouge1', 0),
            'ROUGE-2': metrics.get('rouge2', 0),
            'ROUGE-L': metrics.get('rougeL', 0),
            'ROUGE-Lsum': metrics.get('rougeLsum', 0),
        }
        table_data.append(row)
    
    if not table_data:
        print("No data for best results table")
        return
    
    df = pd.DataFrame(table_data)
    
    # Sort by ROUGE-L descending (since that's what we used for best)
    df = df.sort_values('ROUGE-L', ascending=False)
    
    # Save as CSV
    csv_path = output_folder / 'best_results_table.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved best results table to {csv_path}")
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with color coding
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best scores in each column
    for col_idx, col_name in enumerate(df.columns):
        if col_name in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']:
            best_val = df[col_name].max()
            for row_idx, val in enumerate(df[col_name]):
                if val == best_val:
                    table[(row_idx + 1, col_idx)].set_facecolor('#90EE90')
                    table[(row_idx + 1, col_idx)].set_text_props(weight='bold')
    
    plt.title(f'{title_prefix}Best Epoch Validation Results (by ROUGE-L)', fontsize=14, weight='bold', pad=20)
    plt.savefig(output_folder / 'best_results_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved best results table image to {output_folder}")
    
    return df


def create_heatmap(results_dict, output_folder, title_prefix=""):
    """
    Create a heatmap showing all metrics across all runs.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect data for all epochs
    all_epochs_data = defaultdict(lambda: defaultdict(dict))
    
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        for epoch, metrics in data['validation'].items():
            for metric_name, value in metrics.items():
                all_epochs_data[metric_name][run_name][f'Epoch {epoch}'] = value
    
    # Create heatmap for each metric
    for metric_name, runs_epochs_data in all_epochs_data.items():
        df = pd.DataFrame(runs_epochs_data).T
        
        if df.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 0.8), max(8, len(df) * 0.5)))
        sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, 
                    cbar_kws={'label': f'{metric_name.upper()} Score'})
        ax.set_title(f'{title_prefix}{metric_name.upper()} Heatmap - All Runs & Epochs', 
                     fontsize=14, weight='bold', pad=20)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Run', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_folder / f'heatmap_{metric_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved heatmaps to {output_folder}")


def create_boxplots_final(results_dict, output_folder, title_prefix=""):
    """
    Create boxplots for final epoch results across all runs.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect all final epoch scores
    metrics_collection = defaultdict(list)
    
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        last_epoch = max(data['validation'].keys())
        for metric_name, value in data['validation'][last_epoch].items():
            metrics_collection[metric_name].append(value)
    
    if not metrics_collection:
        print("No data for boxplot")
        return
    
    # Prepare data for boxplot
    metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']
    data_to_plot = [metrics_collection.get(m, []) for m in metric_names]
    
    # Filter out empty metrics
    filtered_data = []
    filtered_labels = []
    for data, label in zip(data_to_plot, metric_labels):
        if data:
            filtered_data.append(data)
            filtered_labels.append(label)
    
    if not filtered_data:
        print("No valid metrics for boxplot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    bp = ax.boxplot(filtered_data, 
                     labels=filtered_labels,
                     patch_artist=True,
                     showmeans=True,
                     meanline=True,
                     notch=False,
                     widths=0.6)
    
    # Customize colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for patch, color in zip(bp['boxes'], colors[:len(filtered_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize whiskers, caps, medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, linestyle='-', color='gray')
    for cap in bp['caps']:
        cap.set(linewidth=1.5, color='gray')
    for median in bp['medians']:
        median.set(linewidth=2, color='darkred')
    for mean in bp['means']:
        mean.set(linewidth=2, color='blue', linestyle='--')
    
    # Labels and title
    ax.set_xlabel('ROUGE Metrics', fontsize=14, weight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, weight='bold')
    ax.set_title(f'{title_prefix}Final Epoch Results Distribution', fontsize=16, weight='bold', pad=20)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend for median and mean
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_folder / 'final_results_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"Saved final results boxplot to {output_folder}")
    plt.close()


def create_boxplots_best(results_dict, output_folder, title_prefix="", metric_for_best='rougeL'):
    """
    Create boxplots for best epoch results (based on highest metric) across all runs.
    
    Args:
        metric_for_best: str - metric to use for determining best epoch (default: 'rougeL')
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect best epoch scores
    metrics_collection = defaultdict(list)
    
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        # Find epoch with best metric score
        best_epoch = None
        best_score = -1
        for epoch, metrics in data['validation'].items():
            score = metrics.get(metric_for_best, 0)
            if score > best_score:
                best_score = score
                best_epoch = epoch
        
        if best_epoch is None:
            continue
            
        # Collect metrics from best epoch
        for metric_name, value in data['validation'][best_epoch].items():
            metrics_collection[metric_name].append(value)
    
    if not metrics_collection:
        print("No data for best results boxplot")
        return
    
    # Prepare data for boxplot
    metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']
    data_to_plot = [metrics_collection.get(m, []) for m in metric_names]
    
    # Filter out empty metrics
    filtered_data = []
    filtered_labels = []
    for data, label in zip(data_to_plot, metric_labels):
        if data:
            filtered_data.append(data)
            filtered_labels.append(label)
    
    if not filtered_data:
        print("No valid metrics for best results boxplot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    bp = ax.boxplot(filtered_data, 
                     labels=filtered_labels,
                     patch_artist=True,
                     showmeans=True,
                     meanline=True,
                     notch=False,
                     widths=0.6)
    
    # Customize colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for patch, color in zip(bp['boxes'], colors[:len(filtered_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize whiskers, caps, medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, linestyle='-', color='gray')
    for cap in bp['caps']:
        cap.set(linewidth=1.5, color='gray')
    for median in bp['medians']:
        median.set(linewidth=2, color='darkred')
    for mean in bp['means']:
        mean.set(linewidth=2, color='blue', linestyle='--')
    
    # Labels and title
    ax.set_xlabel('ROUGE Metrics', fontsize=14, weight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, weight='bold')
    ax.set_title(f'{title_prefix}Best Epoch Results Distribution (by ROUGE-L)', fontsize=16, weight='bold', pad=20)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend for median and mean
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_folder / 'best_results_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"Saved best results boxplot to {output_folder}")
    plt.close()


def create_boxplots_all_epochs(results_dict, output_folder, title_prefix=""):
    """
    Create boxplots for ALL epochs across ALL runs.
    Shows the complete distribution of scores throughout training.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect ALL epoch scores from ALL runs
    metrics_collection = defaultdict(list)
    
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        # Collect metrics from ALL epochs
        for epoch, metrics in data['validation'].items():
            for metric_name, value in metrics.items():
                metrics_collection[metric_name].append(value)
    
    if not metrics_collection:
        print("No data for all epochs boxplot")
        return
    
    # Prepare data for boxplot
    metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']
    data_to_plot = [metrics_collection.get(m, []) for m in metric_names]
    
    # Filter out empty metrics
    filtered_data = []
    filtered_labels = []
    for data, label in zip(data_to_plot, metric_labels):
        if data:
            filtered_data.append(data)
            filtered_labels.append(label)
    
    if not filtered_data:
        print("No valid metrics for all epochs boxplot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    bp = ax.boxplot(filtered_data, 
                     labels=filtered_labels,
                     patch_artist=True,
                     showmeans=True,
                     meanline=True,
                     notch=False,
                     widths=0.6)
    
    # Customize colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for patch, color in zip(bp['boxes'], colors[:len(filtered_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize whiskers, caps, medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, linestyle='-', color='gray')
    for cap in bp['caps']:
        cap.set(linewidth=1.5, color='gray')
    for median in bp['medians']:
        median.set(linewidth=2, color='darkred')
    for mean in bp['means']:
        mean.set(linewidth=2, color='blue', linestyle='--')
    
    # Labels and title
    ax.set_xlabel('ROUGE Metrics', fontsize=14, weight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, weight='bold')
    ax.set_title(f'{title_prefix}All Epochs Results Distribution', fontsize=16, weight='bold', pad=20)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend for median and mean
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add subtitle with sample count
    total_samples = len(filtered_data[0]) if filtered_data else 0
    fig.text(0.5, 0.02, f'Total data points: {total_samples} (all epochs from all runs)', 
             ha='center', fontsize=10, style='italic', color='gray')
    
    # Tight layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_folder / 'all_epochs_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"Saved all epochs boxplot to {output_folder}")
    plt.close()


def create_summary_statistics(results_dict, output_folder):
    """
    Create summary statistics across all runs.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect all final epoch scores
    metrics_collection = defaultdict(list)
    
    for run_name, data in results_dict.items():
        if not data['validation']:
            continue
        
        last_epoch = max(data['validation'].keys())
        for metric_name, value in data['validation'][last_epoch].items():
            metrics_collection[metric_name].append(value)
    
    # Calculate statistics
    stats_data = []
    for metric_name, values in metrics_collection.items():
        stats_data.append({
            'Metric': metric_name.upper(),
            'Mean': round(np.mean(values), 2),
            'Std': round(np.std(values), 2),
            'Min': round(np.min(values), 4),
            'Max': round(np.max(values), 4),
            'Median': round(np.median(values), 4),
            'Count': len(values)
        })
    
    df = pd.DataFrame(stats_data)
    
    # Save as CSV
    csv_path = output_folder / 'summary_statistics.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved summary statistics to {csv_path}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Summary Statistics Across All Runs', fontsize=14, weight='bold', pad=20)
    plt.savefig(output_folder / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def main():
    """
    Main function to generate all visualizations.
    """
    # Define paths - modify these as needed
    results_folders = {
        'YAKE+DBPEDIA': {
            'path': r'',
            'recursive': False
        },
        'GoldSack': {
            'path': r'',
            'recursive': True
        },
        # Add more folders here if needed, e.g.:
        # 'BART': {'path': r'path\to\BART\results', 'recursive': False},
        # 'PubMed': {'path': r'path\to\PubMed\results', 'recursive': False},
    }
    
    output_base = r''
    
    for model_name, config in results_folders.items():
        folder_path = config['path']
        recursive = config.get('recursive', False)
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
        
        # Collect results
        results = collect_all_results(folder_path, recursive=recursive)
        
        if not results:
            print(f"No results found in {folder_path}")
            continue
        
        print(f"Found {len(results)} result files")
        
        # Simplify run names to run_1, run_2, etc.
        # Sort by original name for consistency
        sorted_runs = sorted(results.items(), key=lambda x: x[0])
        simplified_results = {}
        for idx, (original_name, data) in enumerate(sorted_runs, start=1):
            simplified_name = f"run_{idx}"
            simplified_results[simplified_name] = data
            print(f"  {original_name} -> {simplified_name}")
        
        results = simplified_results
        
        # Create output folder for this model
        output_folder = Path(output_base) / model_name
        
        # Generate all visualizations
        print("\nGenerating visualizations...")
        plot_training_curves(results, output_folder / 'training_curves', title_prefix=f"{model_name} - ")
        plot_validation_evolution(results, output_folder / 'validation_evolution', title_prefix=f"{model_name} - ")
        plot_final_comparison_bars(results, output_folder / 'comparisons', title_prefix=f"{model_name} - ")
        plot_final_comparison_bars(
            results,
            output_folder / 'comparisons',
            title_prefix=f"{model_name} - ",
            only_epochs=20,
            output_suffix="_epoch20"
        )
        create_results_table(results, output_folder / 'tables', title_prefix=f"{model_name} - ")
        create_best_results_table(results, output_folder / 'tables', title_prefix=f"{model_name} - ")
        create_boxplots_final(results, output_folder / 'tables', title_prefix=f"{model_name} - ")
        create_boxplots_best(results, output_folder / 'tables', title_prefix=f"{model_name} - ")
        create_boxplots_all_epochs(results, output_folder / 'tables', title_prefix=f"{model_name} - ")
        create_heatmap(results, output_folder / 'heatmaps', title_prefix=f"{model_name} - ")
        create_summary_statistics(results, output_folder / 'statistics')
        
        print(f"\nAll visualizations saved to: {output_folder}")
    
    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
