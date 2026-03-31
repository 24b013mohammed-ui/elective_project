"""
Evaluation module for model performance analysis and ablation studies.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import calculate_metrics, log_metrics, calculate_feature_importance_from_ablation
from model import FinancialSpectrogramCNN
from training import train_model, load_checkpoint, test_model
import config

logger = logging.getLogger(__name__)


def evaluate_model(model, test_loader, loss_fn, scalers_dict, device):
    """
    Comprehensive model evaluation on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        loss_fn: Loss function
        scalers_dict (dict): Dictionary of scalers for each ticker (to inverse-transform predictions)
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: Comprehensive evaluation metrics and predictions with metadata
    """
    logger.info("Starting comprehensive model evaluation...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Extract dates and tickers directly from dataset (before batching)
    # This ensures metadata is preserved
    try:
        if hasattr(test_loader, 'dataset'):
            all_dates = list(test_loader.dataset.dates) if hasattr(test_loader.dataset, 'dates') else []
            all_tickers = list(test_loader.dataset.tickers) if hasattr(test_loader.dataset, 'tickers') else []
            logger.info(f"Extracted {len(all_dates)} dates and {len(all_tickers)} tickers from dataset")
        else:
            all_dates = []
            all_tickers = []
            logger.warning("No dataset attribute in test_loader")
    except Exception as e:
        logger.warning(f"Error extracting metadata from dataset: {e}")
        all_dates = []
        all_tickers = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Unpack batch (spectrogram, target) - metadata removed by collate_fn
            spectrograms, targets = batch_data
            
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)
            
            predictions = model(spectrograms)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    
    logger.info(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
    logger.info(f"Metadata - Dates: {len(all_dates)}, Tickers: {len(all_tickers)}")
    
    # Calculate metrics in normalized space
    metrics_normalized = calculate_metrics(targets, predictions)
    log_metrics(metrics_normalized, "Test (Normalized)")
    
    # Inverse transform to original price scale (if scalers available)
    # CRITICAL FIX: Always preserve original predictions for error calculation
    predictions_original = predictions.copy()  # Default to normalized if inverse transform fails
    targets_original = targets.copy()
    
    try:
        if scalers_dict and all_tickers and all_dates:
            # Validate metadata lengths match predictions
            if len(all_tickers) == len(predictions) and len(all_dates) == len(predictions):
                predictions_original_list = []
                targets_original_list = []
                
                for i, ticker in enumerate(all_tickers):
                    if i < len(predictions) and ticker in scalers_dict:
                        scaler = scalers_dict[ticker]
                        
                        # Inverse transform
                        from utils import inverse_normalize_price
                        pred_orig = inverse_normalize_price([predictions[i]], scaler, feature_index=3)
                        target_orig = inverse_normalize_price([targets[i]], scaler, feature_index=3)
                        
                        predictions_original_list.append(pred_orig[0])
                        targets_original_list.append(target_orig[0])
                
                # Only use inverse-transformed data if we got enough results (at least 80% success)
                if len(predictions_original_list) >= len(predictions) * 0.8:
                    predictions_original = np.array(predictions_original_list)
                    targets_original = np.array(targets_original_list)
                    
                    metrics_original = calculate_metrics(targets_original, predictions_original)
                    log_metrics(metrics_original, "Test (Original Scale)")
                    logger.info(f"Successfully inverse-transformed {len(predictions_original)}/{len(predictions)} predictions using scalers")
                else:
                    logger.warning(f"Inverse transformation had low success rate ({len(predictions_original_list)}/{len(predictions)}), using normalized values")
                    metrics_original = metrics_normalized
                    predictions_original = predictions
                    targets_original = targets
            else:
                logger.warning(f"Metadata length mismatch: tickers={len(all_tickers)}, dates={len(all_dates)}, predictions={len(predictions)}")
                metrics_original = metrics_normalized
        else:
            logger.info(f"Skipping inverse transformation - scalers={bool(scalers_dict)}, tickers={len(all_tickers)}, dates={len(all_dates)}")
            metrics_original = metrics_normalized
    except Exception as e:
        logger.warning(f"Error in inverse-transformation block: {e}")
        metrics_original = metrics_normalized
        predictions_original = predictions
        targets_original = targets
    
    # Ensure all data is numpy arrays before subtraction
    predictions_original_arr = np.array(predictions_original) if not isinstance(predictions_original, np.ndarray) else predictions_original
    targets_original_arr = np.array(targets_original) if not isinstance(targets_original, np.ndarray) else targets_original
    
    # Calculate errors - simple and reliable
    try:
        errors_normalized = (targets - predictions).tolist()
        errors_original = (targets_original_arr - predictions_original_arr).tolist()
        
        # Fallback if errors_original is empty
        if len(errors_original) == 0:
            logger.warning("errors_original empty, using normalized")
            errors_original = errors_normalized
            
        logger.info(f"Errors calculated: normalized={len(errors_normalized)}, original={len(errors_original)}")
    except Exception as e:
        logger.error(f"Error in error calculation: {e}")
        errors_normalized = (targets - predictions).tolist()
        errors_original = errors_normalized
    
    results = {
        'metrics_normalized': metrics_normalized,
        'metrics_original': metrics_original,
        'predictions_normalized': predictions,
        'targets_normalized': targets,
        'predictions_original': predictions_original_arr.tolist() if isinstance(predictions_original_arr, np.ndarray) else predictions_original,
        'targets_original': targets_original_arr.tolist() if isinstance(targets_original_arr, np.ndarray) else targets_original,
        'errors_normalized': errors_normalized,
        'errors_original': errors_original,
        'dates': all_dates,
        'tickers': all_tickers
    }
    
    logger.info(f"Evaluation complete - {len(all_dates)} dates, {len(all_tickers)} tickers, {len(errors_original)} errors")
    return results


def feature_ablation_study(train_loader, val_loader, test_loader, device, 
                           epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE):
    """
    Perform ablation study by removing one channel at a time and measuring performance drop.
    
    Args:
        train_loader, val_loader, test_loader: DataLoaders
        device: 'cuda' or 'cpu'
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        dict: Feature importance scores
    """
    logger.info("Starting feature ablation study...")
    
    baseline_model = FinancialSpectrogramCNN(num_channels=5).to(device)
    baseline_model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
    
    baseline_loss = test_model(baseline_model, test_loader, nn.MSELoss(), device)
    logger.info(f"Baseline test loss: {baseline_loss:.6f}")
    
    feature_importance = {}
    
    # Ablate each channel
    for ablate_idx in range(5):
        logger.info(f"Ablating channel {ablate_idx}...")
        
        ablated_model = FinancialSpectrogramCNN(num_channels=5, ablate_channel=ablate_idx).to(device)
        ablated_model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
        
        ablated_loss = test_model(ablated_model, test_loader, nn.MSELoss(), device)
        
        importance = (ablated_loss - baseline_loss) / baseline_loss
        feature_importance[f'channel_{ablate_idx}'] = importance
        
        logger.info(f"Channel {ablate_idx} importance: {importance:.6f}")
    
    logger.info(f"Feature importance: {feature_importance}")
    return feature_importance
