"""
Financial Time-Series CNN Forecasting Pipeline
5-Day-Ahead Price Prediction using Spectrograms and Convolutional Neural Networks

Phase 1: Data Preparation & Normalization
Phase 2: Signal Processing (STFT Spectrograms)
Phase 3: Model Development (CNN Architecture)
Phase 4: Training (with validation & checkpointing)
Phase 5: Evaluation & Analysis (with feature ablation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys
import numpy as np
from pathlib import Path

# Import project modules
import config
from data_preparation import fetch_and_prepare_data
from dataset_preparation import create_windowed_dataset, train_val_test_split, create_pytorch_dataloaders
from signal_processing import generate_spectrograms
from model import FinancialSpectrogramCNN
from training import train_model, load_checkpoint, test_model
from evaluation import evaluate_model, feature_ablation_study, generate_model_summary
from visualization import (
    generate_all_required_figures, plot_training_curves, plot_predictions_vs_actual,
    plot_residuals
)
from utils import log_metrics

# Create directories BEFORE setting up logging
Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(config.FIGURES_DIR).mkdir(parents=True, exist_ok=True)
Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
checkpoint_dir = os.path.dirname(config.MODEL_CHECKPOINT_PATH)
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

# Setup logging AFTER directories exist
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'run.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary output directories (already created at module level)."""
    pass


def main():
    """Main orchestration function for the entire pipeline."""
    
    logger.info("="*80)
    logger.info("FINANCIAL TIME-SERIES CNN FORECASTING PIPELINE")
    logger.info("5-Day-Ahead Price Prediction")
    logger.info("="*80)
    
    create_directories()
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Config: Horizon={config.PREDICTION_HORIZON}d, Window={config.WINDOW_LENGTH}d, HopSize={config.HOP_SIZE}d")
    
    # ==================== PHASE 1: Data Preparation ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA PREPARATION & NORMALIZATION")
    logger.info("="*80)
    
    try:
        logger.info(f"Fetching data for tickers: {config.TICKERS}")
        data_dict = fetch_and_prepare_data(
            config.TICKERS,
            config.START_DATE,
            config.END_DATE
        )
        logger.info(f"[OK] Successfully loaded data for {len(data_dict)} tickers")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load data: {e}")
        return
    
    # Store scalers for later (for inverse transformation)
    scalers_dict = {ticker: data_dict[ticker]['scaler'] for ticker in data_dict}
    
    # ==================== PHASE 2: Signal Processing ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: SIGNAL PROCESSING (STFT SPECTROGRAMS)")
    logger.info("="*80)
    
    try:
        logger.info("Creating windowed spectrogram dataset...")
        dataset = create_windowed_dataset(
            data_dict,
            window_length=config.WINDOW_LENGTH,
            hop_size=config.HOP_SIZE,
            prediction_horizon=config.PREDICTION_HORIZON
        )
        logger.info(f"[OK] Dataset created: {len(dataset['spectrograms'])} samples")
        
        # Store a sample spectrogram for visualization
        sample_spectrogram = dataset['spectrograms'][0]
        logger.info(f"  Sample spectrogram shape: {sample_spectrogram.shape}")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to create spectrograms: {e}")
        return
    
    # ==================== Phase 2b: Data Splitting ====================
    logger.info("\nSplitting data into train/val/test sets...")
    
    try:
        split_data = train_val_test_split(
            dataset,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO
        )
        logger.info("[OK] Data split complete (temporal order preserved)")
        
        # Create DataLoaders
        dataloaders = create_pytorch_dataloaders(
            split_data,
            batch_size=config.BATCH_SIZE,
            shuffle_train=False  # Preserve temporal order
        )
        logger.info(f"[OK] DataLoaders created")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to split data: {e}")
        return
    
    # ==================== PHASE 3: Model Setup ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: MODEL DEVELOPMENT")
    logger.info("="*80)
    
    try:
        model = FinancialSpectrogramCNN(num_features=config.INPUT_CHANNELS)
        model = model.to(config.DEVICE)
        logger.info(f"[OK] Model created: FinancialSpectrogramCNN")
        # Parameters will be initialized after first forward pass through LazyLinear layer
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to create model: {e}")
        return
    
    # ==================== PHASE 4: Training ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: TRAINING (with Validation & Early Stopping)")
    logger.info("="*80)
    logger.info(f"Epochs: {config.NUM_EPOCHS}, Learning Rate: {config.LEARNING_RATE}, Patience: {config.EARLY_STOPPING_PATIENCE}")
    
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn = nn.MSELoss()
        
        history = train_model(
            model,
            dataloaders['train'],
            dataloaders['val'],
            optimizer,
            loss_fn,
            config.NUM_EPOCHS,
            config.DEVICE,
            checkpoint_dir=os.path.dirname(config.MODEL_CHECKPOINT_PATH),
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )
        
        logger.info(f"[OK] Training complete")
        logger.info(f"  Best epoch: {history['best_epoch']}")
        logger.info(f"  Best validation loss: {history['best_val_loss']:.6f}")
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== PHASE 5: Evaluation & Analysis ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: EVALUATION & ANALYSIS")
    logger.info("="*80)
    
    # Load best model
    logger.info("Loading best model checkpoint...")
    checkpoint = load_checkpoint(model, config.MODEL_CHECKPOINT_PATH, config.DEVICE)
    if checkpoint is None:
        logger.warning("Could not load checkpoint, using current model state")
    
    # Test set evaluation
    logger.info("\nEvaluating on test set...")
    
    try:
        test_results = evaluate_model(
            model,
            dataloaders['test'],
            loss_fn,
            scalers_dict,
            config.DEVICE
        )
        logger.info("[OK] Test evaluation complete")
        
    except Exception as e:
        logger.error(f"[ERROR] Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate summary report
    summary_report = generate_model_summary(test_results)
    logger.info(summary_report)
    
    # Save report to file
    report_path = os.path.join(config.OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(summary_report)
    logger.info(f"Report saved to {report_path}")
    
    # ==================== PHASE 5b: Feature Ablation Study (Optional) ====================
    logger.info("\n" + "="*80)
    logger.info("FEATURE ABLATION STUDY (comparing price-only vs. all features)")
    logger.info("="*80)
    
    # Note: This would ideally use separate datasets with different feature sets
    logger.warning("Note: Ablation study runs on same feature set. Ideally would use separate datasets.")
    
    # Uncomment to run full ablation study (takes additional time)
    # try:
    #     ablation_results = feature_ablation_study(
    #         dataloaders['train'],
    #         dataloaders['val'],
    #         dataloaders['test'],
    #         config.DEVICE
    #     )
    #     logger.info("[OK] Feature ablation study complete")
    # except Exception as e:
    #     logger.warning(f"Feature ablation study skipped: {e}")
    #     ablation_results = None
    
    # For now, set to None (can be uncommented later if needed)
    ablation_results = None
    
    # ==================== Visualization ====================
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    try:
        logger.info("Generating training curves...")
        plot_training_curves(history, output_dir=config.FIGURES_DIR)
        
        logger.info("Generating prediction vs actual plot...")
        plot_predictions_vs_actual(
            test_results['predictions_normalized'],
            test_results['targets_normalized'],
            output_dir=config.FIGURES_DIR
        )
        
        logger.info("Generating residuals plot...")
        plot_residuals(
            test_results['predictions_normalized'],
            test_results['targets_normalized'],
            output_dir=config.FIGURES_DIR
        )
        
        logger.info("Generating all required figures...")
        spectrogram_data = (sample_spectrogram, np.arange(sample_spectrogram.shape[1]), 
                           np.arange(sample_spectrogram.shape[2]))
        
        generate_all_required_figures(
            data_dict,
            spectrogram_data,
            history,
            test_results['predictions_normalized'],
            test_results['targets_normalized'],
            ablation_results=ablation_results,
            output_dir=config.FIGURES_DIR
        )
        
        logger.info(f"[OK] All visualizations saved to {config.FIGURES_DIR}")
        
    except Exception as e:
        logger.error(f"[ERROR] Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== Final Summary ====================
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Figures directory: {config.FIGURES_DIR}")
    logger.info(f"Logs directory: {config.LOGS_DIR}")
    logger.info(f"Model checkpoint: {config.MODEL_CHECKPOINT_PATH}")
    logger.info("\nGenerated Files:")
    for file in sorted(os.listdir(config.FIGURES_DIR)):
        logger.info(f"  - {file}")
    
    logger.info("\n[OK] Financial forecasting pipeline executed successfully!")


if __name__ == "__main__":
    main()