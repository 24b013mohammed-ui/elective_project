"""
Training module for financial CNN model.
Handles training loops, validation, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
from tqdm import tqdm
import config

logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch_num):
    """
    Train for a single epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        loss_fn: Loss function (e.g., MSELoss)
        device: 'cuda' or 'cpu'
        epoch_num (int): Current epoch number (for logging)
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch_num} [Train]", leave=True)
    
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch (spectrogram, target)
        spectrograms, targets = batch_data
        
        # Move to device
        spectrograms = spectrograms.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(spectrograms)
        
        # Calculate loss
        loss = loss_fn(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Epoch {epoch_num} - Training Loss: {epoch_loss:.6f}")
    
    return epoch_loss


def validate_epoch(model, val_loader, loss_fn, device, epoch_num):
    """
    Validate for a single epoch (no gradients computed).
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        device: 'cuda' or 'cpu'
        epoch_num (int): Current epoch number (for logging)
    
    Returns:
        tuple: (average_val_loss, predictions, targets, metadata)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    all_metadata = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch_num} [Val]", leave=True)
        
        for batch_data in pbar:
            # Unpack batch
            if len(batch_data) == 3:
                spectrograms, targets, metadata = batch_data
            else:
                spectrograms, targets = batch_data
                metadata = None
            
            # Move to device
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(spectrograms)
            
            # Calculate loss
            loss = loss_fn(predictions, targets)
            
            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            if metadata:
                all_metadata.append(metadata)
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Epoch {epoch_num} - Validation Loss: {epoch_loss:.6f}")
    
    return epoch_loss, all_predictions, all_targets, all_metadata


def train_model(model, train_loader, val_loader, optimizer, loss_fn, 
                epochs, device, checkpoint_dir=config.MODEL_CHECKPOINT_PATH.rsplit('/', 1)[0],
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE):
    """
    Full training loop with validation, early stopping, and checkpointing.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer instance
        loss_fn: Loss function
        epochs (int): Number of epochs to train
        device: 'cuda' or 'cpu'
        checkpoint_dir (str): Directory to save model checkpoints
        early_stopping_patience (int): Stop if val loss doesn't improve for N epochs
    
    Returns:
        dict: Training history with 'train_loss', 'val_loss', 'best_epoch'
    """
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'patience_counter': 0
    }
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Device: {device}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        val_loss, _, _, _ = validate_epoch(model, val_loader, loss_fn, device, epoch)
        history['val_loss'].append(val_loss)
        
        # Check for improvement
        if val_loss < history['best_val_loss']:
            logger.info(f"✓ Validation loss improved from {history['best_val_loss']:.6f} to {val_loss:.6f}")
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            history['patience_counter'] = 0
            
            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'history': history
            }, checkpoint_path)
            logger.info(f"Model saved to {checkpoint_path}")
        
        else:
            history['patience_counter'] += 1
            logger.warning(f"No improvement for {history['patience_counter']}/{early_stopping_patience} epochs")
        
        # Early stopping check
        if history['patience_counter'] >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Log summary
        logger.info(f"Epoch {epoch}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    logger.info(f"Training complete. Best validation loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']}")
    
    return history


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model to load state into
        checkpoint_path (str): Path to checkpoint file
        device: 'cuda' or 'cpu'
        optimizer: Optional optimizer to restore state
    
    Returns:
        dict: Checkpoint state (contains 'epoch', 'model_state_dict', 'loss', 'history')
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
    return checkpoint


def test_model(model, test_loader, loss_fn, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        loss_fn: Loss function
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: Contains 'test_loss', 'predictions', 'targets', 'metadata'
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    all_metadata = []
    
    logger.info("Running test set evaluation...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=True)
        
        for batch_data in pbar:
            # Unpack batch
            if len(batch_data) == 3:
                spectrograms, targets, metadata = batch_data
            else:
                spectrograms, targets = batch_data
                metadata = None
            
            # Move to device
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(spectrograms)
            
            # Calculate loss
            loss = loss_fn(predictions, targets)
            
            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            if metadata:
                all_metadata.append(metadata)
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    test_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    results = {
        'test_loss': test_loss,
        'predictions': all_predictions,
        'targets': all_targets,
        'metadata': all_metadata
    }
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    return results
