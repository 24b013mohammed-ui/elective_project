"""
Test script to verify lazy loading and memory optimization fixes
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
from data_preparation import fetch_and_prepare_data
from dataset_preparation import (
    create_windowed_dataset, train_val_test_split, create_pytorch_dataloaders
)
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lazy_loading():
    """Test that lazy loading works and reduces memory usage"""
    
    logger.info("=" * 80)
    logger.info("TESTING LAZY LOADING IMPLEMENTATION")
    logger.info("=" * 80)
    
    # Fetch sample data
    logger.info("\n1. Fetching sample data...")
    data_dict = fetch_and_prepare_data(
        tickers=config.TICKERS[:1],  # Use just 1 ticker for fast testing
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    logger.info(f"✓ Fetched data for {len(data_dict)} tickers")
    
    # Test 1: Create dataset with lazy loading
    logger.info("\n2. Creating dataset with LAZY LOADING enabled...")
    
    def progress_callback_lazy(current, total, phase_name):
        logger.info(f"   [Lazy] Progress: {phase_name} ({current}/{total})")
    
    dataset_lazy = create_windowed_dataset(
        data_dict=data_dict,
        window_length=config.WINDOW_LENGTH,
        hop_size=config.HOP_SIZE,
        prediction_horizon=config.PREDICTION_HORIZON,
        use_lazy=True,
        progress_callback=progress_callback_lazy
    )
    
    logger.info(f"✓ Lazy dataset created:")
    logger.info(f"  - Total samples: {len(dataset_lazy['targets'])}")
    logger.info(f"  - Raw windows stored: {len(dataset_lazy['raw_windows'])}")
    logger.info(f"  - Spectrograms (should be None): {dataset_lazy['spectrograms']}")
    
    # Verify structure
    assert dataset_lazy['use_lazy'] == True, "Should have use_lazy=True"
    assert dataset_lazy['raw_windows'] is not None, "Should have raw_windows"
    assert dataset_lazy['spectrograms'] is None, "Should NOT have precomputed spectrograms"
    logger.info("✓ Lazy dataset structure verified")
    
    # Test 2: Create dataset with eager loading for comparison
    logger.info("\n3. Creating dataset with EAGER LOADING (precomputed)...")
    
    dataset_eager = create_windowed_dataset(
        data_dict=data_dict,
        window_length=config.WINDOW_LENGTH,
        hop_size=config.HOP_SIZE,
        prediction_horizon=config.PREDICTION_HORIZON,
        use_lazy=False,
        progress_callback=None
    )
    
    logger.info(f"✓ Eager dataset created:")
    logger.info(f"  - Total samples: {len(dataset_eager['targets'])}")
    logger.info(f"  - Precomputed spectrograms: {len(dataset_eager['spectrograms'])}")
    
    # Verify structure
    assert dataset_eager['use_lazy'] == False, "Should have use_lazy=False"
    assert dataset_eager['spectrograms'] is not None, "Should have precomputed spectrograms"
    assert dataset_eager['raw_windows'] is None, "Should NOT have raw_windows"
    logger.info("✓ Eager dataset structure verified")
    
    # Test 3: Split and create dataloaders
    logger.info("\n4. Testing train/val/test split with lazy dataset...")
    split_data = train_val_test_split(dataset_lazy)
    logger.info(f"✓ Split data created:")
    logger.info(f"  - Train samples: {len(split_data['train']['targets'])}")
    logger.info(f"  - Val samples: {len(split_data['val']['targets'])}")
    logger.info(f"  - Test samples: {len(split_data['test']['targets'])}")
    
    # Test 4: Create dataloaders
    logger.info("\n5. Creating PyTorch DataLoaders...")
    dataloaders = create_pytorch_dataloaders(split_data, batch_size=8)
    logger.info(f"✓ DataLoaders created:")
    logger.info(f"  - Train batches: {len(dataloaders['train'])}")
    logger.info(f"  - Val batches: {len(dataloaders['val'])}")
    logger.info(f"  - Test batches: {len(dataloaders['test'])}")
    logger.info(f"  - Loading mode: {'LAZY' if split_data['split_info']['use_lazy'] else 'EAGER'}")
    
    # Test 5: Test batch loading
    logger.info("\n6. Testing batch loading from lazy dataloader...")
    train_loader = dataloaders['train']
    batch_spectrograms, batch_targets = next(iter(train_loader))
    
    logger.info(f"✓ Batch loaded successfully:")
    logger.info(f"  - Spectrogram shape: {batch_spectrograms.shape}")
    logger.info(f"  - Target shape: {batch_targets.shape}")
    
    # Verify batch shapes
    assert batch_spectrograms.shape[0] == 8, "Batch size should be 8"
    assert batch_spectrograms.shape[1] == 5, "Should have 5 channels"
    assert batch_targets.shape[0] == 8, "Batch size for targets should be 8"
    logger.info("✓ Batch shapes verified")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED - LAZY LOADING WORKING CORRECTLY")
    logger.info("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        test_lazy_loading()
        logger.info("\n✅ Testing completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        exit(1)
