import pandas as pd
import numpy as np
from fetch_data import get_ticker_data 

def engineer_features(df):
    """
    Engineers new features from the historical stock data.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing the historical stock data with columns 'Open', 'High', 'Low', 'Close', 'Volume'.
    
    Returns:
    pd.DataFrame: A DataFrame with new engineered features added.
    """
    # Calculate daily returns
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()

    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() 

    # Calculate moving averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # Momentum signal
    df['Momentum'] = df['MA_10'] / df['MA_50'] 

    # Volume change compared to the 20-day average
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Change'] = (df['Volume'] - df['Volume_MA_20']) / df['Volume_MA_20']

    # Relative Strength Index (RSI)
    gains = df['Daily_Return'].clip(lower=0)
    losses = df['Daily_Return'].clip(upper=0).abs()
    avg_gain = gains.rolling(window=14).mean()
    avg_loss = losses.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)  # Drop rows with NaN values resulting from rolling calculations
    return df

if __name__ == "__main__":
    ticker = 'SPY'
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    
    data = get_ticker_data(ticker, start_date, end_date)
    engineered_data = engineer_features(data)
    print(engineered_data.head())
    print(engineered_data.shape)
    engineered_data.to_csv(f'data/{ticker}_engineered_features.csv', index=True)