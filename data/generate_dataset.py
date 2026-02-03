import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

NSE_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BAJFINANCE", "BHARTIARTL", "KOTAKBANK", "WIPRO", "AXISBANK", "ITC", "HCLTECH",
    "ASIANPAINT", "MARUTI", "TATAMOTORS", "SUNPHARMA", "TITAN", "NTPC", "ULTRACEMCO",
    "TECHM", "GRASIM", "POWERGRID", "COALINDIA", "BPCL", "ONGC", "JSWSTEEL",
    "DIVISLAB", "DRREDDY"
]

def generate_stock_data(symbol: str, days: int = 1095) -> pd.DataFrame:
    """Generate 3 years of synthetic stock data"""
    import hashlib
    hash_obj = hashlib.md5(symbol.encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    np.random.seed(seed)
    
    # Base price range for different stock categories
    if symbol in ["RELIANCE", "TCS", "HDFCBANK"]:
        base_price = np.random.uniform(2000, 3500)
    elif symbol in ["SBIN", "BHARTIARTL", "TATAMOTORS"]:
        base_price = np.random.uniform(500, 1500)
    else:
        base_price = np.random.uniform(100, 2500)
    
    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # Filter out weekends (NSE is closed on weekends)
    dates = dates[dates.weekday < 5]
    days = len(dates)
    
    # Generate price series with realistic patterns
    prices = []
    volumes = []
    
    current_price = base_price
    
    for i in range(days):
        # Market trend (slight upward bias over long term)
        trend = 0.0001 + (i / days) * 0.00005
        
        # Random walk with volatility
        volatility = 0.015 + 0.01 * np.sin(i / 30)  # Varying volatility
        random_change = np.random.normal(trend, volatility)
        
        # Seasonal patterns
        seasonal = 0.002 * np.sin(2 * np.pi * i / 365)
        
        # Momentum effect
        if i > 0:
            momentum = 0.1 * (prices[-1] - base_price) / base_price
        else:
            momentum = 0
        
        # Calculate new price with overflow protection
        total_change = np.clip(random_change + seasonal + momentum, -0.5, 0.5)  # Limit to ±50%
        current_price = current_price * (1 + total_change)
        current_price = max(min(current_price, 100000), 10)  # Reasonable price bounds
        
        prices.append(current_price)
        
        # Generate volume (correlated with price movement)
        base_volume = np.random.uniform(100000, 5000000)
        if i > 0:
            price_movement = abs(prices[-1] - prices[-2]) / prices[-2]
            volume_multiplier = np.clip(1 + price_movement * 10, 0.1, 100)  # Reasonable bounds
        else:
            volume_multiplier = 1
        
        volume = int(np.clip(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0), 1000, 50000000))
        volumes.append(volume)
    
    # Create OHLC data
    data = []
    for i, (date, close_price, volume) in enumerate(zip(dates, prices, volumes)):
        # Generate intraday range
        intraday_volatility = np.random.uniform(0.005, 0.03)
        
        high = close_price * (1 + abs(np.random.normal(0, intraday_volatility)))
        low = close_price * (1 - abs(np.random.normal(0, intraday_volatility)))
        
        # Ensure high >= close >= low
        high = max(high, close_price)
        low = min(low, close_price)
        
        # Open price (previous close with some gap)
        if i == 0:
            open_price = close_price * np.random.uniform(0.98, 1.02)
        else:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        
        open_price = max(min(open_price, high), low)
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)

def generate_dataset():
    """Generate complete dataset for all NSE stocks"""
    print("Generating synthetic dataset for NSE stocks...")
    
    all_data = {}
    
    for symbol in NSE_STOCKS:
        print(f"Generating data for {symbol}...")
        df = generate_stock_data(symbol, days=1095)  # 3 years of data
        all_data[symbol] = df
        
        # Save individual stock data
        df.to_csv(f"data/{symbol}_data.csv", index=False)
    
    # Save metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'total_stocks': len(NSE_STOCKS),
        'data_period_days': 1095,
        'stocks': NSE_STOCKS
    }
    
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset generated successfully for {len(NSE_STOCKS)} stocks")
    print("Data saved in data/ directory")
    
    return all_data

def calculate_dataset_statistics():
    """Calculate and display dataset statistics"""
    stats = {}
    
    for symbol in NSE_STOCKS:
        try:
            df = pd.read_csv(f"data/{symbol}_data.csv")
            stats[symbol] = {
                'mean_price': float(df['close'].mean()),
                'min_price': float(df['close'].min()),
                'max_price': float(df['close'].max()),
                'volatility': float(df['close'].pct_change().std()),
                'total_volume': int(df['volume'].sum()),
                'data_points': int(len(df))
            }
        except FileNotFoundError:
            print(f"Data file not found for {symbol}")
    
    # Save statistics
    with open('data/dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

if __name__ == "__main__":
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Generate dataset
    dataset = generate_dataset()
    
    # Calculate statistics
    stats = calculate_dataset_statistics()
    
    print("\nDataset generation complete!")
    print(f"Total stocks: {len(NSE_STOCKS)}")
    print(f"Data points per stock: ~1095 (3 years)")
    print("Files saved in data/ directory")
