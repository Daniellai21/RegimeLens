import yfinance as yf
import pandas as pd

def get_ticker_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker symbol from Yahoo Finance.
    
    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format
    
    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)  # Flatten the MultiIndex columns
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    ticker = 'SPY'
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    
    data = get_ticker_data(ticker, start_date, end_date)
    print(data.head())
    print(data.shape)
    data.to_csv(f'data/{ticker}_historical_data.csv', index=True)
