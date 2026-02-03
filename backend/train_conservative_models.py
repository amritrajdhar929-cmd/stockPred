import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.conservative_predictor import ConservativePredictor
import joblib
from datetime import datetime
import json

# List of NSE stocks to train
NSE_STOCKS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'ASIANPAINT', 
    'BAJFINANCE', 'HINDUNILVR', 'ITC', 'MARUTI', 'KOTAKBANK', 'WIPRO', 'TITAN', 
    'ULTRACEMCO', 'JSWSTEEL', 'NTPC', 'ONGC', 'POWERGRID', 'GRASIM', 'TECHM', 
    'HCLTECH', 'SUNPHARMA', 'TATAMOTORS', 'AXISBANK', 'M&M', 'BPCL', 'COALINDIA'
]

def train_all_conservative_models():
    """Train conservative models for all NSE stocks"""
    print("🚀 Training Conservative Models for Realistic Predictions...")
    print("=" * 70)
    
    training_results = {}
    models_dir = "backend/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    for i, symbol in enumerate(NSE_STOCKS, 1):
        print(f"\n[{i}/{len(NSE_STOCKS)}] 🎯 Training conservative model for {symbol}...")
        
        try:
            # Create and train conservative predictor
            predictor = ConservativePredictor()
            success = predictor.train_conservative_model(symbol)
            
            if success:
                training_results[symbol] = {
                    'status': 'success',
                    'validation_scores': predictor.validation_scores,
                    'trained_at': datetime.now().isoformat()
                }
                print(f"✅ Conservative model trained successfully for {symbol}")
                
                # Print validation scores
                for timeframe, scores in predictor.validation_scores.items():
                    print(f"   📊 {timeframe}: MAE = ₹{scores['mae']:.2f}, R² = {scores['r2']:.4f}")
            else:
                training_results[symbol] = {
                    'status': 'failed',
                    'error': 'Training failed',
                    'trained_at': datetime.now().isoformat()
                }
                print(f"❌ Conservative training failed for {symbol}")
                
        except Exception as e:
            training_results[symbol] = {
                'status': 'error',
                'error': str(e),
                'trained_at': datetime.now().isoformat()
            }
            print(f"❌ Error training {symbol}: {e}")
    
    # Save training results
    with open('backend/conservative_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in training_results.values() if r['status'] == 'success')
    total = len(training_results)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 Conservative Training Summary:")
    print(f"✅ Successfully trained: {successful}/{total} models")
    print(f"❌ Failed: {total - successful}/{total} models")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Results saved in: backend/conservative_training_results.json")
    
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

def test_conservative_model(symbol: str):
    """Test a trained conservative model"""
    try:
        # Create predictor and load trained model
        predictor = ConservativePredictor()
        if predictor.load_model(symbol):
            # Generate test data
            df = predictor.generate_realistic_data(symbol, days=365)
            current_price = df['close'].iloc[-1]
            
            predictions, confidence = predictor.predict_conservative(df, current_price)
            
            print(f"\n🔍 Conservative predictions for {symbol}:")
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
            print(f"❌ No conservative model found for {symbol}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error testing conservative model for {symbol}: {e}")
        return None, None

if __name__ == "__main__":
    # Train all conservative models
    results = train_all_conservative_models()
    
    # Test a few models
    print(f"\n" + "=" * 70)
    print("🧪 Testing Conservative Models...")
    
    # Test first few stocks
    test_symbols = NSE_STOCKS[:3]
    
    for symbol in test_symbols:
        test_conservative_model(symbol)
        print("-" * 50)
