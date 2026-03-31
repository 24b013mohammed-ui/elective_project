"""
Visualization module for generating all required plots and figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import logging
import os
from pathlib import Path
from scipy.fft import fft, fftfreq
import config

logger = logging.getLogger(__name__)


def setup_output_dir(output_dir=config.FIGURES_DIR):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_training_curves(history, output_dir=config.FIGURES_DIR, filename='training_curves.png'):
    """
    Plot training and validation loss curves over epochs.
    
    Args:
        history (dict): Output from train_model() containing 'train_loss', 'val_loss'
        output_dir (str): Directory to save figure
        filename (str): Filename for figure
    """
    setup_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    
    # Mark best epoch
    best_epoch = history['best_epoch']
    best_loss = history['best_val_loss']
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
    ax.scatter([best_epoch], [best_loss], color='green', s=100, zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Training curves saved to {filepath}")
    plt.close()


def plot_predictions_vs_actual(predictions, targets, dates=None, tickers=None, 
                                output_dir=config.FIGURES_DIR, filename='predictions_vs_actual.png'):
    """
    Plot predicted prices vs actual prices on test set.
    
    Args:
        predictions (np.ndarray): Predicted normalized prices or original prices
        targets (np.ndarray): Actual normalized prices or original prices
        dates (list): Dates corresponding to predictions (optional)
        tickers (list): Ticker names (optional, for legend)
        output_dir (str): Directory to save figure
        filename (str): Filename for figure
    """
    setup_output_dir(output_dir)
    
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_axis = range(len(targets))
    
    ax.plot(x_axis, targets, 'b-o', label='Actual Price', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(x_axis, predictions, 'r--s', label='Predicted Price', linewidth=2, markersize=4, alpha=0.7)
    
    ax.set_xlabel('Test Sample Index', fontsize=12)
    ax.set_ylabel('Normalized Price', fontsize=12)
    ax.set_title('CNN Predictions vs Actual Prices (Test Set, 5-Day Lookahead)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Predictions plot saved to {filepath}")
    plt.close()


def plot_spectrogram(spectrogram,  freq_axis, time_axis, channel_idx=3, channel_name='Close',
                      output_dir=config.FIGURES_DIR, filename='spectrogram_visualization.png'):
    """
    Plot a 2D spectrogram heatmap (single channel).
    
    Args:
        spectrogram (np.ndarray): Shape (num_channels, num_frequencies, num_times)
        freq_axis (np.ndarray): Frequency values
        time_axis (np.ndarray): Time window indices
        channel_idx (int): Which channel to visualize (default 3 = Close price)
        channel_name (str): Name of channel for title
        output_dir (str): Directory to save figure
        filename (str): Filename for figure
    """
    setup_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract single channel spectrogram
    spec_channel = spectrogram[channel_idx, :, :]
    
    # Create heatmap
    im = ax.pcolormesh(time_axis, freq_axis, spec_channel, shading='gouraud', cmap='viridis')
    
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xlabel('Time Windows', fontsize=12)
    ax.set_title(f'Spectrogram: {channel_name} Price\n(Frequency vs Time Energy Content)', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Energy (|FFT|²)', fontsize=11)
    
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Spectrogram saved to {filepath}")
    plt.close()


def plot_time_series(prices, dates=None, ticker_name='Stock',
                      output_dir=config.FIGURES_DIR, filename='time_series.png'):
    """
    Plot raw time series of stock prices.
    
    Args:
        prices (np.ndarray): Price values over time
        dates (list or np.ndarray): Dates corresponding to prices (optional)
        ticker_name (str): Ticker/security name for title
        output_dir (str): Directory to save figure
        filename (str): Filename for figure
    """
    setup_output_dir(output_dir)
    
    prices = np.asarray(prices).flatten()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if dates is None:
        x_axis = range(len(prices))
        ax.plot(x_axis, prices, 'b-', linewidth=1.5)
        ax.set_xlabel('Time Index', fontsize=12)
    else:
        x_axis = range(len(prices))
        ax.plot(x_axis, prices, 'b-', linewidth=1.5)
        ax.set_xlabel('Time (Trading Days)', fontsize=12)
    
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title(f'Time Series: {ticker_name} Historical Prices', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Time series plot saved to {filepath}")
    plt.close()


def plot_frequency_spectrum(signal, sampling_rate=1.0, fft_type='close',
                             output_dir=config.FIGURES_DIR, filename='frequency_spectrum.png'):
    """
    Plot frequency spectrum (FFT) of raw signal.
    
    Args:
        signal (np.ndarray): Time series signal (1D)
        sampling_rate (float): Sampling rate
        fft_type (str): Type of FFT ('close', 'open', etc.) for title
        output_dir (str): Directory to save figure
        filename (str): Filename for figure
    """
    setup_output_dir(output_dir)
    
    signal = np.asarray(signal).flatten()
    
    # Compute FFT
    fft_values = fft(signal)
    freq = fftfreq(len(signal), 1/sampling_rate)
    
    # Only plot positive frequencies
    positive_freq_idx = freq > 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(freq[positive_freq_idx], np.abs(fft_values[positive_freq_idx]), 'purple', linewidth=1.5)
    
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title(f'Frequency Spectrum: {fft_type.capitalize()} Price\n(FFT of Raw Time Series)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Frequency spectrum saved to {filepath}")
    plt.close()


def plot_cnn_architecture(output_dir=config.FIGURES_DIR, filename='cnn_architecture.png'):
    """
    Create a diagram of the CNN architecture.
    
    Args:
        output_dir (str): Directory to save figure
        filename (str): Filename for figure
    """
    setup_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Financial Spectrogram CNN Architecture', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input
    input_box = mpatches.FancyBboxPatch((0.5, 5.5), 1.5, 1, 
                                        boxstyle="round,pad=0.1", 
                                        edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 6, 'Input\n(5,F,T)', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Conv1
    conv1_box = mpatches.FancyBboxPatch((2.5, 5.5), 1.5, 1,
                                         boxstyle="round,pad=0.1",
                                         edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(conv1_box)
    ax.text(3.25, 6, 'Conv2D\n(5→16)', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.arrow(2, 6, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # MaxPool1
    pool1_box = mpatches.FancyBboxPatch((4.5, 5.5), 1.5, 1,
                                        boxstyle="round,pad=0.1",
                                        edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(pool1_box)
    ax.text(5.25, 6, 'MaxPool\n(16)', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.arrow(4, 6, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Conv2
    conv2_box = mpatches.FancyBboxPatch((6.5, 5.5), 1.5, 1,
                                        boxstyle="round,pad=0.1",
                                        edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(conv2_box)
    ax.text(7.25, 6, 'Conv2D\n(16→32)', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.arrow(6, 6, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # MaxPool2
    pool2_box = mpatches.FancyBboxPatch((8.5, 5.5), 1.5, 1,
                                        boxstyle="round,pad=0.1",
                                        edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(pool2_box)
    ax.text(9.25, 6, 'MaxPool\n(32)', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.arrow(8, 6, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Flatten
    flat_box = mpatches.FancyBboxPatch((10.5, 5.5), 1.5, 1,
                                       boxstyle="round,pad=0.1",
                                       edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(flat_box)
    ax.text(11.25, 6, 'Flatten', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.arrow(10, 6, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # FC1
    fc1_box = mpatches.FancyBboxPatch((1, 3.5), 2, 1,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='black', facecolor='lightsalmon', linewidth=2)
    ax.add_patch(fc1_box)
    ax.text(2, 4, 'FC-128\nReLU', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(2, 5.5), xytext=(2, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(10.5, 5.5), xytext=(6, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # FC2
    fc2_box = mpatches.FancyBboxPatch((4, 3.5), 2, 1,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='black', facecolor='lightsalmon', linewidth=2)
    ax.add_patch(fc2_box)
    ax.text(5, 4, 'FC-64\nReLU', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.arrow(3, 4, 1, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Output
    output_box = mpatches.FancyBboxPatch((6.5, 3.5), 2, 1,
                                         boxstyle="round,pad=0.1",
                                         edgecolor='black', facecolor='lightsteelblue', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7.5, 4, 'Output-1\nLinear', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.arrow(6, 4, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Legend and notes
    legend_y = 2.5
    ax.text(0.5, legend_y, 'Input:', fontsize=10, fontweight='bold')
    ax.text(0.5, legend_y - 0.4, '• Spectrogram: 5 channels (OHLCV)', fontsize=9)
    ax.text(0.5, legend_y - 0.8, '• Freq × Time dimensions', fontsize=9)
    
    ax.text(5.5, legend_y, 'Architecture:', fontsize=10, fontweight='bold')
    ax.text(5.5, legend_y - 0.4, '• 2 Convolutional layers (detect patterns)', fontsize=9)
    ax.text(5.5, legend_y - 0.8, '• Max pooling (dimensionality reduction)', fontsize=9)
    
    ax.text(10, legend_y, 'Output:', fontsize=10, fontweight='bold')
    ax.text(10, legend_y - 0.4, '• Single regression neuron', fontsize=9)
    ax.text(10, legend_y - 0.8, '• Predicts 5-day-ahead price', fontsize=9)
    
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"CNN architecture diagram saved to {filepath}")
    plt.close()


def plot_residuals(predictions, targets, output_dir=config.FIGURES_DIR, filename='residuals.png'):
    """
    Plot prediction residuals (errors) over time.
    
    Args:
        predictions (np.ndarray): Predicted values
        targets (np.ndarray): Actual values
        output_dir (str): Directory to save figure
        filename (str): Filename for figure
    """
    setup_output_dir(output_dir)
    
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    residuals = predictions - targets
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Residuals over time
    x_axis = range(len(residuals))
    ax1.plot(x_axis, residuals, 'r-o', linewidth=1, markersize=4, alpha=0.6)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.fill_between(x_axis, residuals, 0, alpha=0.3, color='red')
    ax1.set_ylabel('Residual (Predicted - Actual)', fontsize=11)
    ax1.set_title('Prediction Residuals Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(residuals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Residual Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Residuals plot saved to {filepath}")
    plt.close()


def plot_ablation_comparison(ablation_results, output_dir=config.FIGURES_DIR, filename='ablation_comparison.png'):
    """
    Plot feature ablation study results comparing price-only vs all-features models.
    
    Args:
        ablation_results (dict): Output from feature_ablation_study()
        output_dir (str): Directory to save figure
        filename (str): Filename for figure
    """
    setup_output_dir(output_dir)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract results
    mse_price_only = ablation_results.get('price_only', {}).get('mse', 0)
    mse_all_features = ablation_results.get('all_features', {}).get('mse', 0)
    
    feature_imp = ablation_results.get('feature_importance', {})
    improvement_pct = feature_imp.get('improvement_pct', 0)
    
    # Bar chart comparing MSE
    models = ['Price Only\n(1 channel)', 'All Features\n(5 channels)']
    mse_values = [mse_price_only, mse_all_features]
    colors = ['#ff9999', '#66b3ff']
    
    bars = ax1.bar(models, mse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=11)
    ax1.set_title('Feature Ablation: Model Comparison\n(Lower MSE is Better)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mse_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Improvement metric
    if mse_price_only > 0:
        improvement = mse_price_only - mse_all_features
        ax2.barh(['Improvement'], [improvement], color=['green' if improvement > 0 else 'red'], 
                 alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_xlabel('MSE Reduction', fontsize=11)
        ax2.set_title(f'Improvement with All Features\n({improvement_pct:.2f}% better)', 
                      fontsize=12, fontweight='bold')
        
        # Add value label
        ax2.text(improvement/2, 0, f'{improvement:.6f}\n({improvement_pct:.2f}%)', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Ablation comparison plot saved to {filepath}")
    plt.close()


def generate_all_required_figures(data_dict, spectrogram_data, history, 
                                   test_predictions, test_targets, 
                                   ablation_results=None,
                                   output_dir=config.FIGURES_DIR):
    """
    Generate all 5 required figures for the assignment.
    
    Args:
        data_dict (dict): Original/normalized data from data_preparation
        spectrogram_data (tuple): (spectrogram, freq_axis, time_axis)
        history (dict): Training history
        test_predictions (np.ndarray): Test set predictions
        test_targets (np.ndarray): Test set targets
        ablation_results (dict): Optional ablation study results
        output_dir (str): Output directory
    """
    logger.info(f"Generating all required figures in {output_dir}...")
    
    setup_output_dir(output_dir)
    
    # 1. Time series plot
    if data_dict:
        first_ticker = list(data_dict.keys())[0]
        prices = data_dict[first_ticker]['original_data']['Close'].values if 'original_data' in data_dict[first_ticker] else data_dict[first_ticker]['data']['Close'].values
        plot_time_series(prices, ticker_name=first_ticker, 
                        output_dir=output_dir, filename='1_time_series.png')
    
    # 2. Frequency spectrum
    if data_dict:
        signal = data_dict[first_ticker]['original_data']['Close'].values if 'original_data' in data_dict[first_ticker] else data_dict[first_ticker]['data']['Close'].values
        plot_frequency_spectrum(signal, 
                               output_dir=output_dir, filename='2_frequency_spectrum.png')
    
    # 3. Spectrogram
    if spectrogram_data is not None:
        spec, freq, times = spectrogram_data
        plot_spectrogram(spec, freq, times, 
                        output_dir=output_dir, filename='3_spectrogram.png')
    
    # 4. CNN architecture diagram
    plot_cnn_architecture(output_dir=output_dir, filename='4_cnn_architecture.png')
    
    # 5. Predictions vs actual
    plot_predictions_vs_actual(test_predictions, test_targets, 
                               output_dir=output_dir, filename='5_predictions_vs_actual.png')
    
    # Additional plots
    plot_training_curves(history, 
                        output_dir=output_dir, filename='6_training_curves.png')
    plot_residuals(test_predictions, test_targets, 
                   output_dir=output_dir, filename='7_residuals.png')
    
    if ablation_results:
        plot_ablation_comparison(ablation_results, 
                                output_dir=output_dir, filename='8_ablation_comparison.png')
    
    logger.info(f"All figures generated successfully in {output_dir}")
