"""
Flask API wrapper for Financial CNN Forecasting Pipeline
Exposes backend functions as REST endpoints for React frontend consumption
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import base64
import logging
import psutil
from datetime import datetime
from pathlib import Path
from threading import Thread

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Global state management
class PipelineState:
    def __init__(self):
        self.status = "idle"  # idle, training, completed, error
        self.progress = 0
        self.error_message = None
        self.training_history = {}
        self.evaluation_results = {}
        self.last_update = None
        self.model = None
        self.scalers = {}
        self.test_loader = None
        self.memory_usage = {}  # Track memory usage

state = PipelineState()

def get_config_val(name, default=None):
    """Lazily load config and get value"""
    try:
        import config
        return getattr(config, name, default)
    except:
        return default

def get_available_memory_mb():
    """Get available system memory in MB"""
    try:
        virtual_mem = psutil.virtual_memory()
        return virtual_mem.available / (1024 * 1024)
    except:
        return 2048  # Default estimate if psutil fails

def log_memory_usage(phase_name):
    """Log current memory usage for debugging"""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        memory_mb = mem_info.rss / (1024 * 1024)
        available_mb = get_available_memory_mb()
        state.memory_usage[phase_name] = {
            'rss_mb': memory_mb,
            'available_mb': available_mb,
            'timestamp': datetime.now().isoformat()
        }
        logger.info(f"[{phase_name}] Memory - Process: {memory_mb:.1f}MB, Available: {available_mb:.1f}MB")
    except Exception as e:
        logger.warning(f"Could not log memory usage: {e}")

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003", "http://localhost:3004", "http://127.0.0.1:3004"]}})

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def inverse_transform_predictions(predictions, targets, ticker, scalers_dict):
    """Convert predictions from normalized to original price scale"""
    if ticker not in scalers_dict:
        return predictions, targets
    
    scaler = scalers_dict[ticker]
    pred_reshaped = predictions.reshape(-1, 1)
    target_reshaped = targets.reshape(-1, 1)
    
    predictions_original = scaler.inverse_transform(pred_reshaped).flatten()
    targets_original = scaler.inverse_transform(target_reshaped).flatten()
    
    return predictions_original, targets_original

def calculate_metrics(predictions, targets):
    """Calculate MSE, MAE, RMSE, MAPE, R²"""
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
    
    # R² Score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2)
    }

def run_training_async(config_overrides):
    """Run full training pipeline in background thread"""
    try:
        import torch
        import config
        from data_preparation import fetch_and_prepare_data
        from dataset_preparation import (
            create_windowed_dataset, train_val_test_split, create_pytorch_dataloaders
        )
        from model import FinancialSpectrogramCNN
        from training import train_model
        from evaluation import evaluate_model

        state.status = "training"
        state.progress = 0
        logger.info(f"Starting training with config: {config_overrides}")
        log_memory_usage("Training Start")
        
        # Check available memory before starting
        available_mb = get_available_memory_mb()
        logger.info(f"Available memory: {available_mb:.1f}MB")
        
        # Phase 1: Data Preparation (20%)
        state.progress = 10
        logger.info("Phase 1: Fetching and preparing data...")
        data_dict = fetch_and_prepare_data(
            tickers=config_overrides.get('tickers', config.TICKERS),
            start_date=config_overrides.get('start_date', config.START_DATE),
            end_date=config_overrides.get('end_date', config.END_DATE)
        )
        state.scalers = {ticker: data_dict[ticker]['scaler'] for ticker in data_dict}
        state.progress = 20
        log_memory_usage("After Data Fetch")
        
        # Phase 2: Signal Processing & Dataset Creation (30%)
        # Use lazy loading if available memory is less than 3GB
        use_lazy_loading = available_mb < 3000
        logger.info(f"Auto-detected lazy loading: {use_lazy_loading} (available_mb={available_mb:.0f})")
        
        def dataset_progress_callback(current, total, phase_name):
            """Callback for dataset creation progress"""
            # Map dataset creation progress from 0 to 5% (from 25% to 30% total)
            progress_pct = 25 + (current / total) * 5 if total > 0 else 25
            state.progress = int(progress_pct)
            logger.info(f"Dataset creation: {phase_name} ({current}/{total}) -> {state.progress}%")
        
        state.progress = 25
        logger.info("Phase 2: Creating windowed dataset (lazy loading enabled)...")
        dataset = create_windowed_dataset(
            data_dict=data_dict,
            window_length=config_overrides.get('window_length', config.WINDOW_LENGTH),
            hop_size=config_overrides.get('hop_size', config.HOP_SIZE),
            prediction_horizon=config_overrides.get('prediction_horizon', config.PREDICTION_HORIZON),
            use_lazy=use_lazy_loading,
            progress_callback=dataset_progress_callback
        )
        state.progress = 30
        log_memory_usage("After Dataset Creation")
        
        # Phase 3: Train/Val/Test Split (40%)
        state.progress = 35
        logger.info("Phase 3: Splitting data into train/val/test...")
        split_data = train_val_test_split(
            dataset=dataset,
            train_ratio=config_overrides.get('train_ratio', config.TRAIN_RATIO),
            val_ratio=config_overrides.get('val_ratio', config.VAL_RATIO),
            test_ratio=config_overrides.get('test_ratio', config.TEST_RATIO)
        )
        
        dataloaders = create_pytorch_dataloaders(
            split_data=split_data,
            batch_size=config_overrides.get('batch_size', config.BATCH_SIZE),
            shuffle_train=False
        )
        state.test_loader = dataloaders['test']
        state.progress = 40
        log_memory_usage("After Data Split & DataLoaders")
        
        # Phase 4: Model Setup & Training (80%)
        state.progress = 45
        logger.info("Phase 4: Setting up model and training...")
        model = FinancialSpectrogramCNN(num_features=5)
        device = config_overrides.get('device', config.DEVICE)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config_overrides.get('learning_rate', config.LEARNING_RATE))
        loss_fn = torch.nn.MSELoss()
        
        training_history = train_model(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=config_overrides.get('num_epochs', config.NUM_EPOCHS),
            device=device,
            checkpoint_dir='checkpoints/',
            early_stopping_patience=config_overrides.get('early_stopping_patience', config.EARLY_STOPPING_PATIENCE)
        )
        state.training_history = training_history
        state.model = model
        state.progress = 80
        log_memory_usage("After Training")
        
        # Phase 5: Evaluation (100%)
        state.progress = 85
        logger.info("Phase 5: Evaluating model...")
        checkpoint = torch.load('checkpoints/best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        eval_results = evaluate_model(
            model=model,
            test_loader=state.test_loader,
            loss_fn=loss_fn,
            scalers_dict=state.scalers,
            device=device
        )
        
        state.evaluation_results = eval_results
        state.progress = 100
        state.status = "completed"
        state.last_update = datetime.now().isoformat()
        log_memory_usage("Training Complete")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        import traceback
        state.status = "error"
        state.error_message = f"{str(e)}"
        state.error_traceback = traceback.format_exc()
        state.progress = 0
        logger.error(f"Training error: {e}", exc_info=True)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200

@app.route('/api/train', methods=['POST'])
def train_endpoint():
    """
    POST /api/train
    Start training pipeline with optional config overrides
    
    Request JSON:
    {
        "tickers": ["RELIANCE.NS", "TCS.NS"],
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "batch_size": 16,
        "num_epochs": 50,
        ...
    }
    
    Response: {"status": "training", "message": "..."}
    """
    if state.status == "training":
        return jsonify({"error": "Training already in progress"}), 409
    
    try:
        config_overrides = request.get_json() or {}
        
        # Start training in background thread
        Thread(target=run_training_async, args=(config_overrides,), daemon=True).start()
        
        return jsonify({
            "status": "training",
            "message": "Training pipeline initiated",
            "tickers": config_overrides.get('tickers', get_config_val('TICKERS')),
            "start_date": config_overrides.get('start_date', get_config_val('START_DATE')),
            "end_date": config_overrides.get('end_date', get_config_val('END_DATE'))
        }), 202  # Accepted
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/pipeline-status', methods=['GET'])
def pipeline_status():
    """
    GET /api/pipeline-status
    Get current training status and progress
    
    Response: {"status": "training|completed|error", "progress": 45, "memory_usage": {...}, ...}
    """
    response = {
        "status": state.status,
        "progress": state.progress,
        "last_update": state.last_update,
        "timestamp": datetime.now().isoformat(),
        "memory_usage": state.memory_usage if state.memory_usage else {}
    }
    
    if state.error_message:
        response["error"] = state.error_message
    
    return jsonify(response), 200

@app.route('/api/training-history', methods=['GET'])
def training_history():
    """
    GET /api/training-history
    Get per-epoch training and validation losses
    
    Response: {"epochs": [1, 2, ...], "train_loss": [...], "val_loss": [...]}
    """
    if not state.training_history:
        return jsonify({"error": "No training history available"}), 404
    
    return jsonify({
        "epochs": list(range(1, len(state.training_history.get('train_loss', [])) + 1)),
        "train_loss": state.training_history.get('train_loss', []),
        "val_loss": state.training_history.get('val_loss', []),
        "best_epoch": state.training_history.get('best_epoch'),
        "best_val_loss": state.training_history.get('best_val_loss'),
        "total_epochs_trained": len(state.training_history.get('train_loss', []))
    }), 200

@app.route('/api/metrics', methods=['GET'])
def metrics_endpoint():
    """
    GET /api/metrics
    Get condensed performance metrics summary
    
    Response: 
    {
        "normalized_scale": {"mse": 0.00287, "mae": 0.0412, ...},
        "original_scale": {"mse": 1245.6, "mae": 28.4, ...}
    }
    """
    try:
        if not state.evaluation_results:
            return jsonify({"error": "No evaluation results available"}), 404
        
        return jsonify({
            "normalized_scale": state.evaluation_results.get('metrics_normalized', {}),
            "original_scale_inr": state.evaluation_results.get('metrics_original', {}),
            "timestamp": state.last_update,
            "total_predictions": len(state.evaluation_results.get('predictions_normalized', []))
        }), 200
    except Exception as e:
        import traceback
        error_msg = f"Error serializing metrics: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"error": str(e), "traceback": error_msg}), 500

@app.route('/api/evaluate', methods=['GET'])
def evaluate_endpoint():
    """
    GET /api/evaluate
    Get full evaluation results including predictions and dates
    
    Response: Full evaluation dict with metrics, predictions, dates, tickers, and errors
    """
    try:
        if not state.evaluation_results:
            return jsonify({"error": "No evaluation results available"}), 404
        
        results = state.evaluation_results.copy()
        
        # Convert numpy arrays to lists for JSON serialization - only if they're not already lists
        if 'predictions_normalized' in results and not isinstance(results['predictions_normalized'], list):
            results['predictions_normalized'] = results['predictions_normalized'].tolist()
        if 'targets_normalized' in results and not isinstance(results['targets_normalized'], list):
            results['targets_normalized'] = results['targets_normalized'].tolist()
        if 'predictions_original' in results and not isinstance(results['predictions_original'], list):
            results['predictions_original'] = results['predictions_original'].tolist()
        if 'targets_original' in results and not isinstance(results['targets_original'], list):
            results['targets_original'] = results['targets_original'].tolist()
        
        # Convert error arrays to lists
        if 'errors_normalized' in results and isinstance(results['errors_normalized'], np.ndarray):
            results['errors_normalized'] = results['errors_normalized'].tolist()
        if 'errors_original' in results and isinstance(results['errors_original'], np.ndarray):
            results['errors_original'] = results['errors_original'].tolist()
        
        # Convert dates to strings
        if 'dates' in results:
            results['dates'] = [str(d) for d in results['dates']]
        
        return jsonify(results), 200
    except Exception as e:
        import traceback
        error_msg = f"Error in /api/evaluate: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"error": str(e), "traceback": error_msg}), 500

@app.route('/api/figures/<figure_name>', methods=['GET'])
def get_figure(figure_name):
    """
    GET /api/figures/<figure_name>
    Serve visualization PNG files
    
    Supported figures: training_curves, predictions_vs_actual, residuals, spectrograms
    """
    valid_figures = ['training_curves', 'predictions_vs_actual', 'residuals', 'spectrograms']
    
    if figure_name not in valid_figures:
        return jsonify({"error": f"Invalid figure. Choose from: {valid_figures}"}), 400
    
    figure_path = Path(f'output/figures/{figure_name}.png')
    
    if not figure_path.exists():
        return jsonify({"error": f"Figure not found: {figure_name}"}), 404
    
    try:
        return send_file(str(figure_path), mimetype='image/png')
    except Exception as e:
        logger.error(f"Error serving figure: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-new', methods=['POST'])
def predict_new():
    """
    POST /api/predict-new
    Run inference on new data for a specific ticker
    
    Request JSON:
    {
        "ticker": "RELIANCE.NS",
        "num_days": 30
    }
    
    Response: {"ticker": "...", "forecast": [...], "prediction_date": "..."}
    """
    if not state.status == "completed":
        return jsonify({"error": "Model not trained yet"}), 400
    
    try:
        import torch
        from data_preparation import fetch_and_prepare_data
        from signal_processing import generate_spectrograms
        
        data = request.get_json()
        ticker = data.get('ticker')
        num_days = data.get('num_days', 5)
        
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400
        
        logger.info(f"predict_new requested for {ticker}. num_days: {num_days}")
        # Fetch recent data (enough to pass 50 days check and get WINDOW_LENGTH data points)
        start_date_str = (pd.Timestamp.now() - pd.Timedelta(days=max(100, num_days + 60))).strftime('%Y-%m-%d')
        end_date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        logger.info(f"Fetching from {start_date_str} to {end_date_str}")
        
        try:
            recent_data = fetch_and_prepare_data(
                tickers=[ticker],
                start_date=start_date_str,
                end_date=end_date_str
            )
        except Exception as ex:
            import traceback
            logger.error(f"fetch_and_prepare_data failed: {ex}\n{traceback.format_exc()}")
            return jsonify({"error": str(ex)}), 400
        
        if ticker not in recent_data:
            return jsonify({"error": f"No data found for ticker: {ticker}"}), 404
        
        # Create spectrograms for prediction
        data_values = recent_data[ticker]['data'].values
        window_len = get_config_val('WINDOW_LENGTH', 20)
        hop_size = get_config_val('HOP_SIZE', 2)

        if len(data_values) < window_len:
            return jsonify({"error": f"Not enough data for {ticker}. Need at least {window_len} days."}), 400
            
        recent_window = data_values[-window_len:]
        _, _, spectrograms = generate_spectrograms(recent_window, window_len=window_len, hop_size=hop_size)
        
        # Run inference
        device = get_config_val('DEVICE', 'cpu')
        spec_tensor = torch.from_numpy(spectrograms).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            prediction_normalized = state.model(spec_tensor).cpu().numpy()
        
        # Inverse transform
        scaler = state.scalers[ticker]
        prediction_original = scaler.inverse_transform(prediction_normalized.reshape(-1, 1)).flatten()
        
        return jsonify({
            "ticker": ticker,
            "forecast": prediction_original.tolist(),
            "forecast_normalized": prediction_normalized.flatten().tolist(),
            "prediction_date": pd.Timestamp.now().strftime('%Y-%m-%d'),
            "prediction_horizon_days": get_config_val('PREDICTION_HORIZON', 5)
        }), 200

        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/reset', methods=['POST'])
def reset_pipeline():
    """
    POST /api/reset
    Reset all state and clear training data
    """
    state.status = "idle"
    state.progress = 0
    state.error_message = None
    state.training_history = {}
    state.evaluation_results = {}
    state.model = None
    state.scalers = {}
    state.test_loader = None
    
    logger.info("Pipeline state reset")
    return jsonify({"message": "Pipeline reset successfully"}), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Create output directories if they don't exist
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    logger.info("🚀 Financial CNN API Server Starting...")
    logger.info(f"Port: 5066")
    logger.info(f"Mode: Lazy Loading (Torch/Config)")
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=5066,
        debug=False,
        use_reloader=False,  # Avoid restarting thread
        threaded=True
    )
