#!/usr/bin/env python
"""Test script to verify DataLoader collation fix"""

import torch
from dataset_preparation import create_pytorch_dataloaders, create_windowed_dataset
from data_preparation import fetch_and_prepare_data
from dataset_preparation import train_val_test_split
import config

print("=" * 60)
print("Testing DataLoader Collation Fix")
print("=" * 60)

try:
    print("\n[1/5] Fetching data...")
    data_dict = fetch_and_prepare_data()
    print(f"✓ Data fetched for {len(data_dict)} tickers")
    
    print("\n[2/5] Creating windowed dataset...")
    dataset_info = create_windowed_dataset(
        data_dict, 
        config.WINDOW_LENGTH, 
        config.HOP_SIZE, 
        config.PREDICTION_HORIZON
    )
    print(f"✓ Created {len(dataset_info['spectrograms'])} samples")
    
    print("\n[3/5] Splitting data...")
    split_data = train_val_test_split(
        dataset_info['spectrograms'],
        dataset_info['targets'],
        dataset_info['dates'],
        dataset_info['tickers'],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print(f"✓ Train: {len(split_data['train']['spectrograms'])}, Val: {len(split_data['val']['spectrograms'])}, Test: {len(split_data['test']['spectrograms'])}")
    
    print("\n[4/5] Creating DataLoaders...")
    dataloaders = create_pytorch_dataloaders(split_data, batch_size=32)
    print("✓ DataLoaders created")
    
    print("\n[5/5] Testing batch loading (should work without collation errors)...")
    for batch_idx, batch_data in enumerate(dataloaders['train']):
        specs, targets = batch_data  # This should work now with custom collate_fn
        print(f"  Batch {batch_idx}:")
        print(f"    Spectrograms shape: {specs.shape} (expected: [batch_size, channels, freq_bins, time_windows])")
        print(f"    Targets shape: {targets.shape} (expected: [batch_size])")
        assert specs.shape[1:] == (5, 128, 18), f"Unexpected spectrogram shape: {specs.shape}"
        assert len(targets.shape) == 1, f"Unexpected targets shape: {targets.shape}"
        if batch_idx >= 2:
            break
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - DataLoader collation is fixed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
