import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.ultra_advanced_predictor import UltraAdvancedPredictor
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

def train_all_ultra_models():
    """Train ultra-advanced ensemble models for all NSE stocks"""
    print("🚀 Training Ultra-Advanced Ensemble Models for NSE Stocks...")
    print("=" * 70)
    
    training_results = {}
    models_dir = "backend/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    for i, symbol in enumerate(NSE_STOCKS, 1):
        print(f"\n[{i}/{len(NSE_STOCKS)}] 🎯 Training ultra-advanced model for {symbol}...")
        
        try:
            # Load historical data
            data_path = f"data/{symbol}_data.csv"
            if not os.path.exists(data_path):
                print(f"❌ Data file not found for {symbol}, skipping...")
                continue
            
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create and train ultra-advanced predictor
            predictor = UltraAdvancedPredictor()
            success = predictor.train_ultra_advanced(symbol, df)
            
            if success:
                training_results[symbol] = {
                    'status': 'success',
                    'data_points': len(df),
                    'validation_scores': predictor.validation_scores,
                    'model_weights': predictor.model_weights,
                    'trained_at': datetime.now().isoformat()
                }
                print(f"✅ Ultra-advanced model trained successfully for {symbol}")
                
                # Print detailed validation scores
                for timeframe, scores in predictor.validation_scores.items():
                    confidence = "High" if scores['r2'] > 0.7 else "Medium" if scores['r2'] > 0.4 else "Low"
                    print(f"   📊 {timeframe}: MAE = ₹{scores['mae']:.2f}, R² = {scores['r2']:.4f}, Confidence = {confidence}")
            else:
                training_results[symbol] = {
                    'status': 'failed',
                    'error': 'Training failed',
                    'trained_at': datetime.now().isoformat()
                }
                print(f"❌ Ultra-advanced training failed for {symbol}")
                
        except Exception as e:
            training_results[symbol] = {
                'status': 'error',
                'error': str(e),
                'trained_at': datetime.now().isoformat()
            }
            print(f"❌ Error training {symbol}: {e}")
    
    # Save training results
    with open('backend/ultra_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in training_results.values() if r['status'] == 'success')
    total = len(training_results)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 Ultra-Advanced Training Summary:")
    print(f"✅ Successfully trained: {successful}/{total} models")
    print(f"❌ Failed: {total - successful}/{total} models")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Results saved in: backend/ultra_training_results.json")
    
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

def test_ultra_model(symbol: str):
    """Test a trained ultra-advanced model"""
    try:
        # Load data
        df = pd.read_csv(f"data/{symbol}_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create predictor and load trained model
        predictor = UltraAdvancedPredictor()
        if predictor.load_model(symbol):
            predictions, confidence = predictor.predict_ultra_advanced(df)
            
            print(f"\n🔍 Ultra-Advanced predictions for {symbol}:")
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
            print(f"❌ No ultra-advanced model found for {symbol}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error testing ultra-advanced model for {symbol}: {e}")
        return None, None

def compare_all_models(symbol: str):
    """Compare all model versions"""
    print(f"\n🔄 Comparing all models for {symbol}...")
    print("-" * 50)
    
    # Test ultra-advanced model
    ultra_pred, ultra_conf = test_ultra_model(symbol)
    
    # Test advanced model
    try:
        from models.advanced_stock_predictor import AdvancedStockPredictor
        df = pd.read_csv(f"data/{symbol}_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        adv_predictor = AdvancedStockPredictor()
        if adv_predictor.load_model(symbol):
            adv_pred, adv_conf = adv_predictor.predict_with_confidence(df)
            print(f"📊 Advanced Model - 1 day: ₹{adv_pred['1_day']:.2f} (conf: {adv_conf['1_day']:.1f}%)")
        else:
            print("❌ Advanced model not found")
    except:
        print("❌ Error with advanced model")
    
    return ultra_pred, ultra_conf

if __name__ == "__main__":
    # Train all ultra-advanced models
    results = train_all_ultra_models()
    
    # Test and compare a few models
    print(f"\n" + "=" * 70)
    print("🧪 Testing Ultra-Advanced Models...")
    
    test_symbols = ["RELIANCE", "TCS", "INFY"]
    for symbol in test_symbols:
        compare_all_models(symbol)
        print("-" * 50)
