"""
Configuration file for Financial Forecasting CNN Project
"""

# ==================== Data Configuration ====================
TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']  # Indian stocks
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# ==================== Signal Processing (STFT) Configuration ====================
WINDOW_LENGTH = 20  # Length of spectrogram window (trading days)
HOP_SIZE = 2        # Stride/hop size for sliding window
PREDICTION_HORIZON = 5  # Predict price 5 days ahead

# ==================== Model Configuration ====================
INPUT_CHANNELS = 5  # Open, High, Low, Close, Volume
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5  # Stop if val loss doesn't improve for N epochs

# ==================== Data Split Configuration ====================
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15   # 15% for validation
TEST_RATIO = 0.15  # 15% for testing

# ==================== Device Configuration ====================
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# ==================== Output Configuration ====================
MODEL_CHECKPOINT_PATH = 'checkpoints/best_model.pt'
OUTPUT_DIR = 'output'
FIGURES_DIR = 'output/figures'
LOGS_DIR = 'output/logs'

# ==================== Feature Ablation Configuration ====================
# Train two models: one with price only, one with all 5 features
ABLATION_VARIANTS = {
    'price_only': ['Close'],
    'all_features': ['Open', 'High', 'Low', 'Close', 'Volume']
}
