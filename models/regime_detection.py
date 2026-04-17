import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from data.features import engineer_features
from data.fetch_data import get_ticker_data

FEATURES = ['Daily_Return', 'Volatility', 'Momentum', 'Volume_Change', 'RSI']

def train_hmm(df):
    """
    Trains a Hidden Markov Model (HMM) to detect market regimes based on engineered features.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing the engineered features for the stock data.
    
    Returns:
    GaussianHMM: A trained HMM model.
    StandardScaler: The scaler used to standardize the features.
    """
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    # Train the HMM
    model = GaussianHMM(n_components=3, covariance_type='full', n_iter=1000)
    model.fit(X)

    return model, scaler

def detect_regimes(df, model, scaler):
    """
    Uses the trained HMM model to detect market regimes in the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing the engineered features for the stock data.
    model (GaussianHMM): A trained HMM model.
    scaler (StandardScaler): The scaler used to standardize the features.
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional 'Regime' column indicating the detected market regime.
    """
    X = scaler.transform(df[FEATURES])

    regimes = model.predict(X)
    df['Regime'] = regimes

    return df 

if __name__ == "__main__":
    ticker = 'SPY'
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    
    data = get_ticker_data(ticker, start_date, end_date)
    engineered_data = engineer_features(data)
    
    model, scaler = train_hmm(engineered_data)
    regime_data = detect_regimes(engineered_data, model, scaler)
    
    print(regime_data.head())
    print(regime_data['Regime'].value_counts())
    print(regime_data.groupby('Regime')[FEATURES].mean())
    regime_data.to_csv(f'data/{ticker}_regime_detection.csv', index=True)