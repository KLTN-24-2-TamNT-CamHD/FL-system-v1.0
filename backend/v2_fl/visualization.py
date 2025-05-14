#!/usr/bin/env python3
"""
Visualization Generator for Federated Learning with GA-Stacking

This script generates visualizations for the federated learning project:
1. Confusion matrix visualization
2. Loss throughout training rounds
3. Comparison table of metrics (F1, Precision, Recall, Accuracy) between clients
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


def load_metrics_data(metrics_dir="metrics/"):
    """
    Load metrics data from the metrics directory
    
    Args:
        metrics_dir: Directory containing metrics files
        
    Returns:
        server_metrics: List of server metrics dictionaries
        client_metrics: Dictionary of client metrics by client ID
    """
    # Find the latest metrics directory
    metrics_path = Path(metrics_dir)
    run_dirs = sorted([d for d in metrics_path.glob("run_*")], reverse=True)
    
    if not run_dirs:
        print(f"No run directories found in {metrics_dir}")
        return None, None
    
    latest_run = run_dirs[0]
    print(f"Loading metrics from {latest_run}")
    
    # Load server metrics
    server_metrics_file = latest_run / "metrics_history.json"
    if not server_metrics_file.exists():
        print(f"Server metrics file not found: {server_metrics_file}")
        server_metrics = None
    else:
        with open(server_metrics_file, 'r') as f:
            server_metrics = json.load(f)
    
    # Load client metrics
    client_metrics = {}
    client_dirs = [d for d in Path(metrics_dir).glob("client-*") if d.is_dir()]
    
    for client_dir in client_dirs:
        client_id = client_dir.name
        client_metrics_file = client_dir / "metrics_history.json"
        
        if client_metrics_file.exists():
            with open(client_metrics_file, 'r') as f:
                client_metrics[client_id] = json.load(f)
    
    return server_metrics, client_metrics


def create_confusion_matrix_visualization(server_metrics, output_dir="visualizations"):
    """
    Create a visualization of the confusion matrix from the last round
    
    Args:
        server_metrics: List of server metrics dictionaries
        output_dir: Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the last round metrics
    last_round = server_metrics[-1]
    
    if "eval_metrics" not in last_round:
        print("No evaluation metrics found in the last round")
        return
    
    metrics = last_round["eval_metrics"]
    
    # Extract confusion matrix values
    tp = metrics.get("true_positives", 0)
    fp = metrics.get("false_positives", 0)
    tn = metrics.get("true_negatives", 0)
    fn = metrics.get("false_negatives", 0)
    
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Calculate total samples and percentages
    total = tp + fp + tn + fn
    cm_percentages = cm / total * 100 if total > 0 else cm
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap (green for correct predictions, red for incorrect)
    colors = ['#67001f', '#d6604d', '#4393c3', '#053061']  # Red gradient to Blue gradient
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    
    # Plot confusion matrix
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap=custom_cmap, cbar=True,
                     annot_kws={"size": 16}, linewidths=0.5)
    
    # Plot percentages
    for i in range(2):
        for j in range(2):
            text = f"{cm_percentages[i, j]:.1f}%"
            ax.text(j + 0.5, i + 0.7, text, ha="center", va="center", color="white", fontsize=12)
    
    # Calculate and display metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # Add metrics text
    plt.text(1.6, 0.2, f"Accuracy: {accuracy:.4f}", fontsize=12)
    plt.text(1.6, 0.4, f"Precision: {precision:.4f}", fontsize=12)
    plt.text(1.6, 0.6, f"Recall: {recall:.4f}", fontsize=12)
    plt.text(1.6, 0.8, f"F1 Score: {f1:.4f}", fontsize=12)
    
    # Add labels
    plt.title("Confusion Matrix - Credit Card Fraud Detection", fontsize=20, pad=20)
    plt.ylabel("Actual Label", fontsize=14, labelpad=10)
    plt.xlabel("Predicted Label", fontsize=14, labelpad=10)
    
    # Set tick labels
    ax.set_xticklabels(['Legitimate', 'Fraud'], fontsize=12)
    ax.set_yticklabels(['Legitimate', 'Fraud'], fontsize=12, rotation=0)
    
    # Add annotations for corners
    plt.text(0.5, 0.35, "True Negatives", ha="center", va="center", fontsize=10, color="white")
    plt.text(1.5, 0.35, "False Positives", ha="center", va="center", fontsize=10, color="white")
    plt.text(0.5, 1.35, "False Negatives", ha="center", va="center", fontsize=10, color="white")
    plt.text(1.5, 1.35, "True Positives", ha="center", va="center", fontsize=10, color="white")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix visualization to {output_dir}/confusion_matrix.png")
    plt.close()


def create_loss_visualization(server_metrics, output_dir="visualizations"):
    """
    Create a visualization of the loss throughout training rounds
    
    Args:
        server_metrics: List of server metrics dictionaries
        output_dir: Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract round numbers and loss values
    rounds = []
    train_losses = []
    eval_losses = []
    accuracies = []
    f1_scores = []
    
    for metric in server_metrics:
        round_num = metric.get("round", None)
        
        if round_num is not None:
            rounds.append(round_num)
            
            # Training loss (if available)
            if "metrics" in metric and "loss" in metric["metrics"]:
                train_losses.append(metric["metrics"]["loss"])
            else:
                train_losses.append(None)
            
            # Evaluation loss
            if "eval_loss" in metric:
                eval_losses.append(metric["eval_loss"])
            elif "eval_metrics" in metric and "loss" in metric["eval_metrics"]:
                eval_losses.append(metric["eval_metrics"]["loss"])
            else:
                eval_losses.append(None)
            
            # Accuracy
            if "eval_metrics" in metric and "avg_accuracy" in metric["eval_metrics"]:
                accuracies.append(metric["eval_metrics"]["avg_accuracy"])
            elif "eval_metrics" in metric and "accuracy" in metric["eval_metrics"]:
                accuracies.append(metric["eval_metrics"]["accuracy"])
            else:
                accuracies.append(None)
            
            # F1 Score
            if "eval_metrics" in metric and "global_f1" in metric["eval_metrics"]:
                f1_scores.append(metric["eval_metrics"]["global_f1"])
            elif "eval_metrics" in metric and "avg_f1_score" in metric["eval_metrics"]:
                f1_scores.append(metric["eval_metrics"]["avg_f1_score"])
            else:
                f1_scores.append(None)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot loss
    ax1.plot(rounds, eval_losses, 'o-', color='#FF5733', linewidth=2, label='Evaluation Loss')
    
    if any(x is not None for x in train_losses):
        ax1.plot(rounds, train_losses, 'o-', color='#C70039', linewidth=2, label='Training Loss')
    
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title('Loss Progress During Federated Learning Rounds', fontsize=18, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Plot accuracy and F1 score
    ax2.plot(rounds, accuracies, 'o-', color='#33A1FF', linewidth=2, label='Accuracy')
    ax2.plot(rounds, f1_scores, 'o-', color='#3D9970', linewidth=2, label='F1 Score')
    
    ax2.set_xlabel('Round', fontsize=14)
    ax2.set_ylabel('Score', fontsize=14)
    ax2.set_title('Performance Metrics During Federated Learning Rounds', fontsize=18, pad=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    # Set integer ticks for rounds
    ax2.set_xticks(rounds)
    ax2.set_xticklabels([str(r) for r in rounds])
    
    # Annotate the final values
    if eval_losses and eval_losses[-1] is not None:
        ax1.annotate(f'{eval_losses[-1]:.4f}', 
                    xy=(rounds[-1], eval_losses[-1]), 
                    xytext=(10, 0), 
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold')
    
    if accuracies and accuracies[-1] is not None:
        ax2.annotate(f'{accuracies[-1]:.4f}', 
                    xy=(rounds[-1], accuracies[-1]), 
                    xytext=(10, 0), 
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold')
    
    if f1_scores and f1_scores[-1] is not None:
        ax2.annotate(f'{f1_scores[-1]:.4f}', 
                    xy=(rounds[-1], f1_scores[-1]), 
                    xytext=(10, -15), 
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold')
    
    # Add annotations for key points
    # Find minimum loss and its round
    if eval_losses and any(x is not None for x in eval_losses):
        min_loss = min(x for x in eval_losses if x is not None)
        min_loss_round = rounds[eval_losses.index(min_loss)]
        ax1.annotate(f'Min Loss: {min_loss:.4f}', 
                    xy=(min_loss_round, min_loss), 
                    xytext=(0, -30), 
                    textcoords='offset points',
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'),
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss_metrics.png"), dpi=300, bbox_inches="tight")
    print(f"Saved loss visualization to {output_dir}/training_loss_metrics.png")
    plt.close()


def create_client_comparison_table(client_metrics, server_metrics=None, output_dir="visualizations"):
    """
    Create a visualization comparing metrics between clients
    
    Args:
        client_metrics: Dictionary of client metrics by client ID
        server_metrics: Optional list of server metrics dictionaries
        output_dir: Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the max round number to use for comparison
    max_round = 0
    for client_id, metrics in client_metrics.items():
        for metric in metrics:
            if "round" in metric:
                max_round = max(max_round, metric["round"])
    
    # Create a dictionary to store metrics for each client
    client_comparison = {}
    
    # Extract metrics for each client
    for client_id, metrics in client_metrics.items():
        # Find metrics for the max round
        client_max_round_metrics = None
        
        for metric in metrics:
            if metric.get("round") == max_round:
                client_max_round_metrics = metric
                break
        
        if client_max_round_metrics:
            # Get accuracy
            accuracy = client_max_round_metrics.get("accuracy", 0)
            
            # Get precision, recall, f1_score
            precision = client_max_round_metrics.get("precision", 0)
            recall = client_max_round_metrics.get("recall", 0)
            f1_score = client_max_round_metrics.get("f1_score", 0)
            
            # Get GA-Stacking specific metrics if available
            ga_metrics = client_max_round_metrics.get("ga_stacking_metrics", {})
            diversity_score = ga_metrics.get("diversity_score", 0)
            final_score = ga_metrics.get("final_score", 0)
            
            client_comparison[client_id] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "diversity_score": diversity_score,
                "final_score": final_score
            }
    
    # Extract global metrics from server if available
    global_metrics = None
    if server_metrics:
        for metric in reversed(server_metrics):
            if metric.get("round") == max_round and "eval_metrics" in metric:
                eval_metrics = metric["eval_metrics"]
                global_metrics = {
                    "accuracy": eval_metrics.get("avg_accuracy", 0),
                    "precision": eval_metrics.get("global_precision", 0),
                    "recall": eval_metrics.get("global_recall", 0),
                    "f1_score": eval_metrics.get("global_f1", 0),
                    "diversity_score": 0,  # Not applicable for global model
                    "final_score": 0  # Not applicable for global model
                }
                break
    
    # Add global metrics to comparison if available
    if global_metrics:
        client_comparison["Global Model"] = global_metrics
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(client_comparison, orient='index')
    
    # Sort by F1 score
    df = df.sort_values("f1_score", ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.8 + 2))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Format values
    formatted_df = df.copy()
    for col in ["accuracy", "precision", "recall", "f1_score", "diversity_score"]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
    
    # Create table
    table = ax.table(
        cellText=formatted_df.values,
        rowLabels=formatted_df.index,
        colLabels=formatted_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    # Color mapping for header
    header_colors = ['#D6EAF8'] * len(formatted_df.columns)
    
    # Highlight the model with the best metrics in different colors
    best_metrics = {
        "accuracy": "#D5F5E3",  # Light green
        "precision": "#FCF3CF",  # Light yellow
        "recall": "#FADBD8",     # Light red
        "f1_score": "#E8DAEF",   # Light purple
        "diversity_score": "#D6DBDF",  # Light gray
        "final_score": "#F5CBA7"  # Light orange
    }
    
    # Highlight cells with best values
    for col_idx, col in enumerate(formatted_df.columns):
        if col in best_metrics:
            # Style header
            table[(0, col_idx)].set_facecolor(header_colors[col_idx])
            table[(0, col_idx)].set_text_props(weight='bold')
            
            # Find best value for this column
            best_row_idx = None
            best_val = -float('inf')
            
            for row_idx, client_id in enumerate(formatted_df.index):
                try:
                    val = df.loc[client_id, col]
                    if val > best_val:
                        best_val = val
                        best_row_idx = row_idx
                except Exception:
                    continue
            
            # Style the best value cell
            if best_row_idx is not None:
                table[(best_row_idx + 1, col_idx)].set_facecolor(best_metrics[col])
                table[(best_row_idx + 1, col_idx)].set_text_props(weight='bold')
    
    # Set row heights and column widths
    for row_idx in range(len(formatted_df) + 1):
        table[(row_idx, 0)].set_text_props(weight='bold')
        table.auto_set_column_width([0, 1, 2, 3, 4, 5])
    
    # Add title
    plt.title(f"Client Metrics Comparison (Round {max_round})", fontsize=18, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "client_metrics_comparison.png"), dpi=300, bbox_inches="tight")
    print(f"Saved client comparison visualization to {output_dir}/client_metrics_comparison.png")
    plt.close()


def main():
    """Main function to create all visualizations"""
    # Load metrics data
    server_metrics, client_metrics = load_metrics_data()
    
    if not server_metrics:
        print("No server metrics data found. Cannot create visualizations.")
        return
    
    # Create output directory
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    create_confusion_matrix_visualization(server_metrics, output_dir)
    create_loss_visualization(server_metrics, output_dir)
    
    if client_metrics:
        create_client_comparison_table(client_metrics, server_metrics, output_dir)
    else:
        print("No client metrics data found. Cannot create client comparison visualization.")


if __name__ == "__main__":
    main()