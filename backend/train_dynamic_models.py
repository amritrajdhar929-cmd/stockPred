import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.dynamic_predictor import DynamicPredictor
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

def train_all_dynamic_models():
    """Train dynamic ensemble models for all available stocks"""
    print("🚀 Training Dynamic Ensemble Models for NSE Stocks...")
    print("=" * 70)
    
    # Get stocks dynamically
    stocks = get_available_stocks()
    print(f"Found {len(stocks)} stocks in data directory")
    
    training_results = {}
    models_dir = "backend/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    for i, symbol in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] 🎯 Training dynamic model for {symbol}...")
        
        try:
            # Load historical data
            data_path = f"data/{symbol}_data.csv"
            if not os.path.exists(data_path):
                print(f"❌ Data file not found for {symbol}, skipping...")
                continue
            
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create and train dynamic predictor
            predictor = DynamicPredictor()
            success = predictor.train_dynamic_model(symbol, df)
            
            if success:
                training_results[symbol] = {
                    'status': 'success',
                    'data_points': len(df),
                    'validation_scores': predictor.validation_scores,
                    'model_weights': predictor.model_weights,
                    'config': predictor.config,
                    'trained_at': datetime.now().isoformat()
                }
                print(f"✅ Dynamic model trained successfully for {symbol}")
                
                # Print validation scores
                for timeframe, scores in predictor.validation_scores.items():
                    print(f"   📊 {timeframe}: MAE = ₹{scores['mae']:.2f}, R² = {scores['r2']:.4f}")
            else:
                training_results[symbol] = {
                    'status': 'failed',
                    'error': 'Training failed',
                    'trained_at': datetime.now().isoformat()
                }
                print(f"❌ Dynamic training failed for {symbol}")
                
        except Exception as e:
            training_results[symbol] = {
                'status': 'error',
                'error': str(e),
                'trained_at': datetime.now().isoformat()
            }
            print(f"❌ Error training {symbol}: {e}")
    
    # Save training results
    with open('backend/dynamic_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in training_results.values() if r['status'] == 'success')
    total = len(training_results)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 Dynamic Training Summary:")
    print(f"✅ Successfully trained: {successful}/{total} models")
    print(f"❌ Failed: {total - successful}/{total} models")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Results saved in: backend/dynamic_training_results.json")
    
    # Calculate average confidence scores
    if successful > 0:
        avg_r2_1d = np.mean([r['validation_scores']['1_day']['r2'] 
                            for r in training_results.values() 
                            if r['status'] == 'success'])
        avg_r2_5d = np.mean([r['validation_scores']['5_day']['r2'] 
                            for r in training_results.values() 
                            if r['status'] == 'success'])
        avg_r2_30d = np.mean([r['validation_scores']['30_day']['r2'] 
                             for r in training_results.values() 
                             if r['status'] == 'success'])
        
        print(f"📈 Average R² Scores:")
        print(f"   1-day: {avg_r2_1d:.4f}")
        print(f"   5-day: {avg_r2_5d:.4f}")
        print(f"   30-day: {avg_r2_30d:.4f}")
    
    return training_results

def test_dynamic_model(symbol: str):
    """Test a trained dynamic model"""
    try:
        # Load data
        df = pd.read_csv(f"data/{symbol}_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create predictor and load trained model
        predictor = DynamicPredictor()
        if predictor.load_model(symbol):
            predictions, confidence = predictor.predict_dynamic(df)
            
            print(f"\n🔍 Dynamic predictions for {symbol}:")
            print(f"💰 Current price: ₹{df['close'].iloc[-1]:.2f}")
            print(f"📈 1 day: ₹{predictions['1_day']:.2f} (confidence: {confidence['1_day']:.1f}%)")
            print(f"📈 5 days: ₹{predictions['5_day']:.2f} (confidence: {confidence['5_day']:.1f}%)")
            print(f"📈 30 days: ₹{predictions['30_day']:.2f} (confidence: {confidence['30_day']:.1f}%)")
            
            # Show validation scores
            if predictor.validation_scores:
                print(f"\n📊 Model Performance:")
                for timeframe, scores in predictor.validation_scores.items():
                    print(f"   {timeframe}: MAE = ₹{scores['mae']:.2f}, R² = {scores['r2']:.4f}")
            
            return predictions, confidence
        else:
            print(f"❌ No dynamic model found for {symbol}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error testing dynamic model for {symbol}: {e}")
        return None, None

if __name__ == "__main__":
    # Train all dynamic models
    results = train_all_dynamic_models()
    
    # Test a few models
    print(f"\n" + "=" * 70)
    print("🧪 Testing Dynamic Models...")
    
    # Get first few stocks for testing
    stocks = get_available_stocks()
    test_symbols = stocks[:3] if len(stocks) >= 3 else stocks
    
    for symbol in test_symbols:
        test_dynamic_model(symbol)
        print("-" * 50)
