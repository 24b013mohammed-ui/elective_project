import numpy as np
from scipy.signal import stft

def generate_spectrograms(time_series_data, window_len=64, hop_size=16):
    """
    Converts a multivariate time series into a spectrogram using STFT.
    
    Args:
        time_series_data: Shape (Time_Steps, Num_Features)
        window_len: STFT window length
        hop_size: Hop size for time windows
    
    Returns:
        f: Frequency bins
        t: Time bins
        multichannel_spectrogram: Shape (Channels, Frequencies, Time_Windows)
    """
    # time_series_data shape: (Total_Time_Steps, Num_Features)
    num_features = time_series_data.shape[1]
    spectrograms = []
    
    # Process each feature (e.g., Open, High, Volume) as a separate "channel"
    for i in range(num_features):
        signal = time_series_data[:, i]
        
        # Apply STFT
        # fs=1.0 (normalized frequency), nperseg=window length (L), noverlap=L-H
        f, t, Zxx = stft(signal, fs=1.0, nperseg=window_len, noverlap=(window_len - hop_size))
        
        # Calculate magnitude squared to get the Spectrogram S(t,f)
        S = np.abs(Zxx)**2 
        spectrograms.append(S)
        
    # Stack features into channels: Shape becomes (Channels, Frequencies, Time_Windows)
    # This directly maps to a CNN input image format
    multichannel_spectrogram = np.stack(spectrograms, axis=0)
    
    return f, t, multichannel_spectrogram