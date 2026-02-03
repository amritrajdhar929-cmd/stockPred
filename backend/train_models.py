import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.stock_predictor import StockPredictor
import joblib
from datetime import datetime
import json

NSE_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BAJFINANCE", "BHARTIARTL", "KOTAKBANK", "WIPRO", "AXISBANK", "ITC", "HCLTECH",
    "ASIANPAINT", "MARUTI", "TATAMOTORS", "SUNPHARMA", "TITAN", "NTPC", "ULTRACEMCO",
    "TECHM", "GRASIM", "POWERGRID", "COALINDIA", "BPCL", "ONGC", "JSWSTEEL",
    "DIVISLAB", "DRREDDY"
]

def train_all_models():
    """Train prediction models for all NSE stocks"""
    print("Training AI models for NSE stocks...")
    
    training_results = {}
    models_dir = "backend/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    for symbol in NSE_STOCKS:
        print(f"\nTraining model for {symbol}...")
        
        try:
            # Load historical data
            data_path = f"data/{symbol}_data.csv"
            if not os.path.exists(data_path):
                print(f"Data file not found for {symbol}, skipping...")
                continue
            
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create and train predictor
            predictor = StockPredictor()
            predictor.train(symbol, df)
            
            if predictor.is_trained:
                training_results[symbol] = {
                    'status': 'success',
                    'data_points': len(df),
                    'trained_at': datetime.now().isoformat()
                }
                print(f"✓ Model trained successfully for {symbol}")
            else:
                training_results[symbol] = {
                    'status': 'failed',
                    'error': 'Training failed',
                    'trained_at': datetime.now().isoformat()
                }
                print(f"✗ Training failed for {symbol}")
                
        except Exception as e:
            training_results[symbol] = {
                'status': 'error',
                'error': str(e),
                'trained_at': datetime.now().isoformat()
            }
            print(f"✗ Error training {symbol}: {e}")
    
    # Save training results
    with open('backend/training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in training_results.values() if r['status'] == 'success')
    total = len(training_results)
    
    print(f"\nTraining Summary:")
    print(f"Successfully trained: {successful}/{total} models")
    print(f"Failed: {total - successful}/{total} models")
    print(f"Models saved in: {models_dir}")
    
    return training_results

def test_model(symbol: str):
    """Test a trained model with sample predictions"""
    try:
        # Load data
        df = pd.read_csv(f"data/{symbol}_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create predictor and load trained model
        predictor = StockPredictor()
        if predictor.load_model(symbol):
            predictions, confidence = predictor.predict(df)
            
            print(f"\nTest predictions for {symbol}:")
            print(f"Current price: ₹{df['close'].iloc[-1]:.2f}")
            print(f"1 day: ₹{predictions['1_day']:.2f} (confidence: {confidence['1_day']:.1f}%)")
            print(f"5 days: ₹{predictions['5_day']:.2f} (confidence: {confidence['5_day']:.1f}%)")
            print(f"30 days: ₹{predictions['30_day']:.2f} (confidence: {confidence['30_day']:.1f}%)")
            
            return predictions, confidence
        else:
            print(f"No trained model found for {symbol}")
            return None, None
            
    except Exception as e:
        print(f"Error testing model for {symbol}: {e}")
        return None, None

if __name__ == "__main__":
    # Train all models
    results = train_all_models()
    
    # Test a few models
    print("\n" + "="*50)
    print("Testing sample models...")
    
    test_symbols = ["RELIANCE", "TCS", "INFY"]
    for symbol in test_symbols:
        test_model(symbol)
