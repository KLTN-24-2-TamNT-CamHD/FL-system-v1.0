"""
Fraud-specific metrics for evaluating model performance in credit card fraud detection.
"""

import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix
)
from typing import Dict, Tuple, List, Union, Any

def calculate_fraud_metrics(
    y_true: Union[np.ndarray, torch.Tensor], 
    y_pred: Union[np.ndarray, torch.Tensor],
    y_score: Union[np.ndarray, torch.Tensor] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate fraud-specific metrics for model evaluation.
    
    Args:
        y_true: Ground truth labels (0 = legitimate, 1 = fraud)
        y_pred: Predicted labels (after thresholding for binary classification)
        y_score: Raw prediction scores (probabilities) for AUC calculation
        threshold: Classification threshold for converting scores to binary predictions
        
    Returns:
        Dictionary with calculated metrics
    """
    # Convert torch tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor) and y_score is not None:
        y_score = y_score.detach().cpu().numpy()
    
    # Ensure correct shapes
    y_true = y_true.flatten()
    if y_pred.shape != y_true.shape:
        y_pred = y_pred.flatten()
    
    # If we have raw scores instead of binary predictions, apply threshold
    if y_score is not None and y_pred is None:
        y_score = y_score.flatten()
        y_pred = (y_score > threshold).astype(int)
    
    # Calculate confusion matrix values
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Handle potential division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy calculation
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        # Calculate AUC only if we have probability scores
        if y_score is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_score)
                metrics['auc_pr'] = average_precision_score(y_true, y_score)  # Precision-Recall AUC
            except ValueError as e:
                # This can happen if we have only one class in y_true
                metrics['auc_roc'] = 0.5
                metrics['auc_pr'] = np.mean(y_true)
        
        return metrics
        
    except ValueError as e:
        # Handle error case
        print(f"Error calculating metrics: {e}")
        return {
            'accuracy': np.mean(y_true == y_pred),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.5,
        }

def calculate_metrics_from_batches(
    data_loader: torch.utils.data.DataLoader, 
    model: torch.nn.Module,
    threshold: float = 0.5,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Calculate fraud metrics over all batches in a data loader.
    
    Args:
        data_loader: PyTorch DataLoader with test/validation data
        model: Model to evaluate
        threshold: Classification threshold
        device: Device to use for computation
        
    Returns:
        Dictionary with calculated metrics
    """
    all_y_true = []
    all_y_pred = []
    all_y_score = []
    
    device = torch.device(device)
    model.eval()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Get model predictions
            output = model(data)
            
            # Make sure output is the right shape and type
            if not isinstance(output, torch.Tensor):
                output = torch.tensor(output, dtype=torch.float32, device=device)
            
            # For binary classification with one output neuron
            if output.shape != target.shape and output.shape[1] == 1:
                output = output.squeeze(1)
            
            # Save raw scores for AUC calculation
            all_y_score.append(output.cpu())
            
            # Convert to binary predictions
            pred = (output > threshold).float()
            
            all_y_true.append(target.cpu())
            all_y_pred.append(pred.cpu())
    
    # Concatenate batch results
    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()
    y_score = torch.cat(all_y_score).numpy()
    
    # Calculate and return metrics
    return calculate_fraud_metrics(y_true, y_pred, y_score, threshold)