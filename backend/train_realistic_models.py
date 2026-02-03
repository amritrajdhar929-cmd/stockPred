import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.realistic_predictor import RealisticPredictor
import joblib
from datetime import datetime
import json

# Load stocks dynamically from data directory
def get_available_stocks():
    """Get list of available stocks from data directory"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    
    stocks = []
    for file in os.listdir(data_dir):
        if file.endswith('_data.csv'):
            symbol = file.replace('_data.csv', '')
            stocks.append(symbol)
    
    return sorted(stocks)

def train_all_realistic_models():
    """Train realistic models for all available stocks"""
    print("🚀 Training Realistic Models for Meaningful Predictions...")
    print("=" * 70)
    
    # Get stocks dynamically
    stocks = get_available_stocks()
    print(f"Found {len(stocks)} stocks in data directory")
    
    training_results = {}
    models_dir = "backend/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    for i, symbol in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] 🎯 Training realistic model for {symbol}...")
        
        try:
            # Load historical data
            data_path = f"data/{symbol}_data.csv"
            if not os.path.exists(data_path):
                print(f"❌ Data file not found for {symbol}, skipping...")
                continue
            
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create and train realistic predictor
            predictor = RealisticPredictor()
            success = predictor.train_realistic_model(symbol, df)
            
            if success:
                training_results[symbol] = {
                    'status': 'success',
                    'data_points': len(df),
                    'validation_scores': predictor.validation_scores,
                    'trained_at': datetime.now().isoformat()
                }
                print(f"✅ Realistic model trained successfully for {symbol}")
                
                # Print validation scores
                for timeframe, scores in predictor.validation_scores.items():
                    print(f"   📊 {timeframe}: MAE = ₹{scores['mae']:.2f}, R² = {scores['r2']:.4f}")
            else:
                training_results[symbol] = {
                    'status': 'failed',
                    'error': 'Training failed',
                    'trained_at': datetime.now().isoformat()
                }
                print(f"❌ Realistic training failed for {symbol}")
                
        except Exception as e:
            training_results[symbol] = {
                'status': 'error',
                'error': str(e),
                'trained_at': datetime.now().isoformat()
            }
            print(f"❌ Error training {symbol}: {e}")
    
    # Save training results
    with open('backend/realistic_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in training_results.values() if r['status'] == 'success')
    total = len(training_results)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 Realistic Training Summary:")
    print(f"✅ Successfully trained: {successful}/{total} models")
    print(f"❌ Failed: {total - successful}/{total} models")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Results saved in: backend/realistic_training_results.json")
    
    # Calculate average performance
    if successful > 0:
        avg_mae_1d = np.mean([r['validation_scores']['1d']['mae'] 
                             for r in training_results.values() 
                             if r['status'] == 'success' and '1d' in r['validation_scores']])
        avg_r2_1d = np.mean([r['validation_scores']['1d']['r2'] 
                            for r in training_results.values() 
                            if r['status'] == 'success' and '1d' in r['validation_scores']])
        
        print(f"📈 Average Performance (1-day):")
        print(f"   MAE = ₹{avg_mae_1d:.2f}, R² = {avg_r2_1d:.4f}")
    
    return training_results

def test_realistic_model(symbol: str):
    """Test a trained realistic model"""
    try:
        # Load data
        df = pd.read_csv(f"data/{symbol}_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Create predictor and load trained model
        predictor = RealisticPredictor()
        if predictor.load_model(symbol):
            predictions, confidence = predictor.predict_realistic(df, current_price)
            
            print(f"\n🔍 Realistic predictions for {symbol}:")
            print(f"💰 Current price: ₹{current_price:.2f}")
            print(f"📈 1 day: ₹{predictions['1d_day']:.2f} (confidence: {confidence['1d_day']:.1f}%)")
            print(f"📈 5 days: ₹{predictions['5d_day']:.2f} (confidence: {confidence['5d_day']:.1f}%)")
            print(f"📈 30 days: ₹{predictions['30d_day']:.2f} (confidence: {confidence['30d_day']:.1f}%)")
            
            # Show validation scores
            if predictor.validation_scores:
                print(f"\n📊 Model Performance:")
                for timeframe, scores in predictor.validation_scores.items():
                    print(f"   {timeframe}: MAE = ₹{scores['mae']:.2f}, R² = {scores['r2']:.4f}")
            
            return predictions, confidence
        else:
            print(f"❌ No realistic model found for {symbol}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error testing realistic model for {symbol}: {e}")
        return None, None

if __name__ == "__main__":
    # Train all realistic models
    results = train_all_realistic_models()
    
    # Test a few models
    print(f"\n" + "=" * 70)
    print("🧪 Testing Realistic Models...")
    
    # Get first few stocks for testing
    stocks = get_available_stocks()
    test_symbols = stocks[:3] if len(stocks) >= 3 else stocks
    
    for symbol in test_symbols:
        test_realistic_model(symbol)
        print("-" * 50)
