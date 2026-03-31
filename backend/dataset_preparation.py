"""
Dataset preparation module for creating windowed spectrograms and data splits.
Handles temporal splitting and PyTorch DataLoader creation.
Supports both eager (precomputed) and lazy (on-demand) spectrogram loading.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from signal_processing import generate_spectrograms
import config

logger = logging.getLogger(__name__)


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for spectrogram-target pairs (EAGER LOADING).
    
    Stores precomputed spectrograms in memory.
    Each sample consists of:
    - Input: Spectrogram of multivariate signal (5 channels, freq_bins, time_windows)
    - Target: Normalized Close price 5 days in the future
    """
    
    def __init__(self, spectrograms, targets, dates, tickers, window_length):
        """
        Args:
            spectrograms (list): List of precomputed spectrogram tensors
            targets (list): List of target prices (normalized)
            dates (list): List of corresponding dates
            tickers (list): List of tickers for each sample
            window_length (int): Window length used for spectrograms
        """
        self.spectrograms = spectrograms
        self.targets = targets
        self.dates = dates
        self.tickers = tickers
        self.window_length = window_length
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        """
        Returns (spectrogram_tensor, target_price)
        Note: Metadata is excluded here to avoid DataLoader collation issues.
        Metadata can be accessed directly if needed for logging/reporting.
        """
        spectrogram = torch.tensor(self.spectrograms[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return spectrogram, target


class LazySpectrogramDataset(Dataset):
    """
    PyTorch Dataset for spectrogram-target pairs (LAZY LOADING).
    
    Computes spectrograms on-demand during training batches.
    This dramatically reduces memory footprint (stores raw windows only).
    Each sample consists of:
    - Input: Raw windowed data, spectrogram computed on-the-fly
    - Target: Normalized Close price 5 days in the future
    """
    
    def __init__(self, raw_windows, targets, dates, tickers, window_length, hop_size):
        """
        Args:
            raw_windows (list): List of raw windowed data arrays (not spectrograms)
            targets (list): List of target prices (normalized)
            dates (list): List of corresponding dates
            tickers (list): List of tickers for each sample
            window_length (int): Window length for STFT computation
            hop_size (int): Hop size for STFT computation
        """
        self.raw_windows = raw_windows
        self.targets = targets
        self.dates = dates
        self.tickers = tickers
        self.window_length = window_length
        self.hop_size = hop_size
        
    def __len__(self):
        return len(self.raw_windows)
    
    def __getitem__(self, idx):
        """
        Computes and returns spectrogram on-demand.
        Returns (spectrogram_tensor, target_price)
        """
        # Get raw window data
        window_data = self.raw_windows[idx]  # Shape: (window_length, 5)
        
        # Compute spectrogram on-the-fly
        try:
            _, _, spectrogram = generate_spectrograms(
                window_data, 
                self.window_length, 
                self.hop_size
            )
            spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"Error computing spectrogram for sample {idx}: {e}")
            # Return zeros if computation fails
            spectrogram_tensor = torch.zeros((5, 33, 4), dtype=torch.float32)
        
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return spectrogram_tensor, target


def create_windowed_dataset(data_dict, window_length, hop_size, prediction_horizon, use_lazy=True, progress_callback=None):
    """
    Create windowed dataset with 5-day lookahead targets.
    
    Supports both eager (precomputed spectrograms) and lazy loading (compute on-demand).
    Lazy loading is recommended for memory-constrained systems.
    
    For each trading day t:
    - Input: Spectrogram from days [t-window_length+1 : t]
    - Target: Normalized Close price at day t+prediction_horizon
    
    Args:
        data_dict (dict): Dictionary of normalized data from data_preparation.fetch_and_prepare_data()
        window_length (int): Length of spectrogram window (in trading days)
        hop_size (int): Stride for sliding window
        prediction_horizon (int): Days ahead to predict (5 in our case)
        use_lazy (bool): If True, use lazy loading (compute spectrograms on-demand).
                        If False, precompute all spectrograms (uses more memory but faster training).
                        Default: True (memory-efficient)
        progress_callback (callable): Optional callback function progress_callback(current, total, phase_name)
                                     for reporting progress to UI/logging
    
    Returns:
        dict: Contains 'spectrograms' (or 'raw_windows' if lazy), 'targets', 'dates', 'tickers' lists
              Also includes 'use_lazy' flag to indicate dataset type
    """
    
    all_spectrograms = []
    all_raw_windows = []  # For lazy loading
    all_targets = []
    all_dates = []
    all_tickers = []
    
    # Count total tickers for progress calculation
    valid_tickers = 0
    for ticker, ticker_data in data_dict.items():
        df_scaled = ticker_data['data']
        num_rows = len(df_scaled)
        min_required = window_length + prediction_horizon
        if num_rows >= min_required:
            valid_tickers += 1
    
    current_ticker_idx = 0
    
    for ticker, ticker_data in data_dict.items():
        df_scaled = ticker_data['data']  # Normalized data
        
        num_rows = len(df_scaled)
        logger.info(f"Processing {ticker}: {num_rows} trading days available")
        
        # Validate data sufficiency
        min_required = window_length + prediction_horizon
        if num_rows < min_required:
            logger.warning(
                f"{ticker} has only {num_rows} days, but needs {min_required} "
                f"(window_length={window_length} + prediction_horizon={prediction_horizon}). "
                f"Skipping this ticker."
            )
            continue
        
        current_ticker_idx += 1
        
        # Slide window through the data
        # We can create samples for time steps: [window_length, ..., num_rows - prediction_horizon)
        batch_size = max(1, (num_rows - window_length - prediction_horizon) // hop_size)
        
        for t_idx, t in enumerate(range(window_length, num_rows - prediction_horizon, hop_size)):
            try:
                # Extract window of multivariate data: [t-window_length : t]
                window_data = df_scaled.iloc[t - window_length : t].values  # Shape: (window_length, 5)
                
                if use_lazy:
                    # Lazy loading: store raw window data, compute spectrogram later
                    all_raw_windows.append(window_data.copy())  # Make a copy to avoid reference issues
                else:
                    # Eager loading: compute spectrogram now
                    f, t_spec, spectrogram = generate_spectrograms(window_data, window_length, hop_size)
                    all_spectrograms.append(spectrogram)
                
                # Get target: Close price at t + prediction_horizon
                target_price = df_scaled.iloc[t + prediction_horizon]['Close']
                target_date = df_scaled.index[t + prediction_horizon]
                
                all_targets.append([target_price])  # Wrap in list to match expected shape
                all_dates.append(target_date)
                all_tickers.append(ticker)
                
                # Report progress
                if progress_callback:
                    total_work = valid_tickers
                    progress = current_ticker_idx + (t_idx / batch_size) if batch_size > 0 else current_ticker_idx
                    progress_callback(progress, total_work, f"Processing {ticker}")
                
            except Exception as e:
                logger.warning(f"Error creating sample at {ticker} index {t}: {e}")
                continue
        
        ticker_samples = len([x for x in all_tickers if x == ticker])
        logger.info(f"Created {ticker_samples} samples from {ticker}")
        
        # Final progress update for this ticker
        if progress_callback:
            progress_callback(current_ticker_idx, valid_tickers, f"Completed {ticker}")
    
    if not all_targets:
        raise ValueError("No valid samples could be created from the input data.")
    
    logger.info(f"Total dataset size: {len(all_targets)} samples")
    logger.info(f"Dataset mode: {'LAZY LOADING (memory-efficient)' if use_lazy else 'EAGER LOADING (precomputed)'}")
    
    return {
        'spectrograms': all_spectrograms if not use_lazy else None,
        'raw_windows': all_raw_windows if use_lazy else None,
        'targets': all_targets,
        'dates': all_dates,
        'tickers': all_tickers,
        'use_lazy': use_lazy,
        'window_length': window_length,
        'hop_size': hop_size
    }


def train_val_test_split(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Chronological (time-aware) split into train/val/test sets.
    Important: Split AFTER windowing to avoid data leakage.
    
    Split order: Train first 70% → Val next 15% → Test last 15%
    This respects temporal causality (no future data in training).
    
    Supports both eager (precomputed spectrograms) and lazy (raw windows) datasets.
    
    Args:
        dataset (dict): Output from create_windowed_dataset()
        train_ratio (float): Proportion for training
        val_ratio (float): Proportion for validation
        test_ratio (float): Proportion for testing
    
    Returns:
        dict: Contains 'train', 'val', 'test' datasets with their indices
    """
    
    n_samples = len(dataset['targets'])
    use_lazy = dataset.get('use_lazy', False)
    window_length = dataset.get('window_length', config.WINDOW_LENGTH)
    hop_size = dataset.get('hop_size', config.HOP_SIZE)
    
    # Calculate split indices (chronological order)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n_samples))
    
    logger.info(f"Chronological split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} samples")
    
    # Create split datasets
    def create_split_data(indices, use_lazy=False):
        split = {
            'targets': [dataset['targets'][i] for i in indices],
            'dates': [dataset['dates'][i] for i in indices],
            'tickers': [dataset['tickers'][i] for i in indices],
            'indices': indices  # Store original indices for reference
        }
        
        if use_lazy:
            split['raw_windows'] = [dataset['raw_windows'][i] for i in indices]
        else:
            split['spectrograms'] = [dataset['spectrograms'][i] for i in indices]
        
        return split
    
    return {
        'train': create_split_data(train_idx, use_lazy),
        'val': create_split_data(val_idx, use_lazy),
        'test': create_split_data(test_idx, use_lazy),
        'split_info': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'total_samples': n_samples,
            'use_lazy': use_lazy,
            'window_length': window_length,
            'hop_size': hop_size
        }
    }


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles batching of (spectrogram, target) tuples.
    
    Args:
        batch (list): List of (spectrogram, target) tuples from dataset
    
    Returns:
        tuple: (tensor of spectrograms, tensor of targets)
    """
    spectrograms, targets = zip(*batch)
    
    # Stack along batch dimension
    spectrograms = torch.stack(spectrograms)
    targets = torch.stack(targets)
    
    return spectrograms, targets


def create_pytorch_dataloaders(split_data, batch_size=config.BATCH_SIZE, shuffle_train=False):
    """
    Create PyTorch DataLoaders from split data.
    
    Supports both eager (precomputed spectrograms) and lazy (on-demand computation) datasets.
    
    Args:
        split_data (dict): Output from train_val_test_split()
        batch_size (int): Batch size for training
        shuffle_train (bool): Whether to shuffle training data
                             Default: False to maintain temporal order (best practice for time-series)
                             Set to True only if you want to trade causality for potential generalization gains
    
    Returns:
        dict: Contains 'train', 'val', 'test' DataLoaders and 'datasets' info
    """
    
    use_lazy = split_data.get('split_info', {}).get('use_lazy', False)
    window_length = split_data.get('split_info', {}).get('window_length', config.WINDOW_LENGTH)
    hop_size = split_data.get('split_info', {}).get('hop_size', config.HOP_SIZE)
    
    # Create PyTorch datasets based on load strategy
    if use_lazy:
        # Lazy loading datasets
        train_dataset = LazySpectrogramDataset(
            split_data['train']['raw_windows'],
            split_data['train']['targets'],
            split_data['train']['dates'],
            split_data['train']['tickers'],
            window_length=window_length,
            hop_size=hop_size
        )
        
        val_dataset = LazySpectrogramDataset(
            split_data['val']['raw_windows'],
            split_data['val']['targets'],
            split_data['val']['dates'],
            split_data['val']['tickers'],
            window_length=window_length,
            hop_size=hop_size
        )
        
        test_dataset = LazySpectrogramDataset(
            split_data['test']['raw_windows'],
            split_data['test']['targets'],
            split_data['test']['dates'],
            split_data['test']['tickers'],
            window_length=window_length,
            hop_size=hop_size
        )
        logger.info("Using LAZY LOADING datasets (spectrograms computed on-demand during training)")
    else:
        # Eager loading datasets
        train_dataset = SpectrogramDataset(
            split_data['train']['spectrograms'],
            split_data['train']['targets'],
            split_data['train']['dates'],
            split_data['train']['tickers'],
            window_length=window_length
        )
        
        val_dataset = SpectrogramDataset(
            split_data['val']['spectrograms'],
            split_data['val']['targets'],
            split_data['val']['dates'],
            split_data['val']['tickers'],
            window_length=window_length
        )
        
        test_dataset = SpectrogramDataset(
            split_data['test']['spectrograms'],
            split_data['test']['targets'],
            split_data['test']['dates'],
            split_data['test']['tickers'],
            window_length=window_length
        )
        logger.info("Using EAGER LOADING datasets (spectrograms precomputed)")
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle test
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        collate_fn=custom_collate_fn
    )
    
    logger.info(
        f"DataLoaders created: "
        f"train_batches={len(train_loader)}, "
        f"val_batches={len(val_loader)}, "
        f"test_batches={len(test_loader)}"
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        },
        'config': {
            'use_lazy': use_lazy,
            'batch_size': batch_size,
            'window_length': window_length,
            'hop_size': hop_size
        }
    }
