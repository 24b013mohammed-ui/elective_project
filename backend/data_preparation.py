import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_and_prepare_data(tickers, start_date, end_date):
    """
    Downloads, aligns, and normalizes financial time series data.
    
    Args:
        tickers (list): List of ticker symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        dict: Dictionary with ticker as key, containing 'data' (normalized DataFrame) and 'scaler' object
    
    Raises:
        ValueError: If ticker data is empty or insufficient
    """
    data_dict = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Downloading data for {ticker}...")
            # Download data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                logger.warning(f"No data available for {ticker}. Skipping.")
                continue
            
            # Select our multivariate signal: [Open, High, Low, Close, Volume]
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[features]
            
            # Validate data
            if len(df) < 50:  # Need minimum data for meaningful analysis
                logger.warning(f"{ticker} has only {len(df)} days. Need at least 50 days. Skipping.")
                continue
            
            # Align: Forward fill any missing values (like holidays)
            df = df.ffill()  # Fixed deprecation: use ffill() instead of fillna(method='ffill')
            
            # Remove any remaining NaN values (for safety)
            df = df.dropna()
            
            if df.empty:
                logger.warning(f"All data was NaN for {ticker}. Skipping.")
                continue
            
            # Normalize the data between 0 and 1 (separate scaler per ticker)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)
            df_scaled = pd.DataFrame(scaled_data, columns=features, index=df.index)
            
            logger.info(f"Successfully prepared {ticker}: {len(df_scaled)} rows")
            
            data_dict[ticker] = {
                'data': df_scaled,
                'scaler': scaler,  # Keep the scaler to reverse-transform predictions later
                'original_data': df  # Store original unscaled data for reference
            }
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not data_dict:
        raise ValueError("No valid data could be downloaded for any ticker.")
    
    logger.info(f"Successfully loaded data for {len(data_dict)} tickers")
    return data_dict