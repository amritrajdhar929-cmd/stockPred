import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.advanced_stock_predictor import AdvancedStockPredictor
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

def train_all_advanced_models():
    """Train advanced XGBoost models for all NSE stocks"""
    print("🚀 Training Advanced XGBoost Models for NSE Stocks...")
    print("=" * 60)
    
    training_results = {}
    models_dir = "backend/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    for i, symbol in enumerate(NSE_STOCKS, 1):
        print(f"\n[{i}/{len(NSE_STOCKS)}] Training advanced model for {symbol}...")
        
        try:
            # Load historical data
            data_path = f"data/{symbol}_data.csv"
            if not os.path.exists(data_path):
                print(f"❌ Data file not found for {symbol}, skipping...")
                continue
            
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create and train advanced predictor
            predictor = AdvancedStockPredictor()
            success = predictor.train_with_validation(symbol, df)
            
            if success:
                training_results[symbol] = {
                    'status': 'success',
                    'data_points': len(df),
                    'validation_scores': predictor.validation_scores,
                    'trained_at': datetime.now().isoformat()
                }
                print(f"✅ Advanced model trained successfully for {symbol}")
                
                # Print validation scores
                for timeframe, scores in predictor.validation_scores.items():
                    print(f"   {timeframe}: MAE = {scores['mae']:.2f} (±{scores['mae_std']:.2f})")
            else:
                training_results[symbol] = {
                    'status': 'failed',
                    'error': 'Training failed',
                    'trained_at': datetime.now().isoformat()
                }
                print(f"❌ Advanced training failed for {symbol}")
                
        except Exception as e:
            training_results[symbol] = {
                'status': 'error',
                'error': str(e),
                'trained_at': datetime.now().isoformat()
            }
            print(f"❌ Error training {symbol}: {e}")
    
    # Save training results
    with open('backend/advanced_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in training_results.values() if r['status'] == 'success')
    total = len(training_results)
    
    print(f"\n" + "=" * 60)
    print(f"🎯 Advanced Training Summary:")
    print(f"✅ Successfully trained: {successful}/{total} models")
    print(f"❌ Failed: {total - successful}/{total} models")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Results saved in: backend/advanced_training_results.json")
    
    return training_results

def test_advanced_model(symbol: str):
    """Test a trained advanced model with sample predictions"""
    try:
        # Load data
        df = pd.read_csv(f"data/{symbol}_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create predictor and load trained model
        predictor = AdvancedStockPredictor()
        if predictor.load_model(symbol):
            predictions, confidence = predictor.predict_with_confidence(df)
            
            print(f"\n🔍 Advanced predictions for {symbol}:")
            print(f"💰 Current price: ₹{df['close'].iloc[-1]:.2f}")
            print(f"📈 1 day: ₹{predictions['1_day']:.2f} (confidence: {confidence['1_day']:.1f}%)")
            print(f"📈 5 days: ₹{predictions['5_day']:.2f} (confidence: {confidence['5_day']:.1f}%)")
            print(f"📈 30 days: ₹{predictions['30_day']:.2f} (confidence: {confidence['30_day']:.1f}%)")
            
            # Show validation scores
            if predictor.validation_scores:
                print(f"\n📊 Validation Performance:")
                for timeframe, scores in predictor.validation_scores.items():
                    print(f"   {timeframe}: MAE = ₹{scores['mae']:.2f}")
            
            return predictions, confidence
        else:
            print(f"❌ No advanced trained model found for {symbol}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error testing advanced model for {symbol}: {e}")
        return None, None

def compare_models(symbol: str):
    """Compare old vs advanced models"""
    print(f"\n🔄 Comparing models for {symbol}...")
    
    # Test old model
    try:
        from models.stock_predictor import StockPredictor
        df = pd.read_csv(f"data/{symbol}_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        old_predictor = StockPredictor()
        if old_predictor.load_model(symbol):
            old_pred, old_conf = old_predictor.predict(df)
            print(f"📊 Old Model - 1 day: ₹{old_pred['1_day']:.2f} (conf: {old_conf['1_day']:.1f}%)")
        else:
            print("❌ Old model not found")
    except:
        print("❌ Error with old model")
    
    # Test advanced model
    adv_pred, adv_conf = test_advanced_model(symbol)
    
    return adv_pred, adv_conf

if __name__ == "__main__":
    # Install XGBoost if not available
    try:
        import xgboost
    except ImportError:
        print("📦 Installing XGBoost...")
        os.system("pip3 install xgboost")
    
    # Train all advanced models
    results = train_all_advanced_models()
    
    # Test and compare a few models
    print(f"\n" + "=" * 60)
    print("🧪 Testing Sample Advanced Models...")
    
    test_symbols = ["RELIANCE", "TCS", "INFY"]
    for symbol in test_symbols:
        compare_models(symbol)
        print("-" * 40)
