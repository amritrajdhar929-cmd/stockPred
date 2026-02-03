import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.optimized_predictor import OptimizedPredictor
import joblib
from datetime import datetime
import json
import time

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

def train_all_optimized_models():
    """Train optimized models for all available stocks"""
    print("🚀 Training Optimized Models for Better Predictions...")
    print("=" * 70)
    
    # Get stocks dynamically
    stocks = get_available_stocks()
    print(f"Found {len(stocks)} stocks in data directory")
    
    training_results = {}
    models_dir = "backend/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    start_time = time.time()
    
    for i, symbol in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] 🎯 Training optimized model for {symbol}...")
        
        try:
            # Load historical data
            data_path = f"data/{symbol}_data.csv"
            if not os.path.exists(data_path):
                print(f"❌ Data file not found for {symbol}, skipping...")
                continue
            
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create and train optimized predictor
            predictor = OptimizedPredictor()
            success = predictor.train_optimized_model(symbol, df)
            
            if success:
                training_results[symbol] = {
                    'status': 'success',
                    'data_points': len(df),
                    'validation_scores': predictor.validation_scores,
                    'model_weights': predictor.model_weights,
                    'config': predictor.config,
                    'trained_at': datetime.now().isoformat()
                }
                print(f"✅ Optimized model trained successfully for {symbol}")
                
                # Print validation scores with confidence estimates
                for timeframe, scores in predictor.validation_scores.items():
                    mae = scores['mae']
                    r2 = scores['r2']
                    confidence = min(95, max(80, 85 + (r2 * 20) - (mae * 0.01)))
                    print(f"   📊 {timeframe}: MAE = ₹{mae:.2f}, R² = {r2:.4f}, Est. Confidence = {confidence:.1f}%")
            else:
                training_results[symbol] = {
                    'status': 'failed',
                    'error': 'Training failed',
                    'trained_at': datetime.now().isoformat()
                }
                print(f"❌ Optimized training failed for {symbol}")
                
        except Exception as e:
            training_results[symbol] = {
                'status': 'error',
                'error': str(e),
                'trained_at': datetime.now().isoformat()
            }
            print(f"❌ Error training {symbol}: {e}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Save training results
    with open('backend/optimized_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in training_results.values() if r['status'] == 'success')
    total = len(training_results)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 Optimized Training Summary:")
    print(f"✅ Successfully trained: {successful}/{total} models")
    print(f"❌ Failed: {total - successful}/{total} models")
    print(f"⏱️ Total training time: {training_time:.2f} seconds")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Results saved in: backend/optimized_training_results.json")
    
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
        
        avg_mae_1d = np.mean([r['validation_scores']['1_day']['mae'] 
                             for r in training_results.values() 
                             if r['status'] == 'success'])
        avg_mae_5d = np.mean([r['validation_scores']['5_day']['mae'] 
                             for r in training_results.values() 
                             if r['status'] == 'success'])
        avg_mae_30d = np.mean([r['validation_scores']['30_day']['mae'] 
                              for r in training_results.values() 
                              if r['status'] == 'success'])
        
        # Estimate average confidence
        avg_conf_1d = min(95, max(80, 85 + (avg_r2_1d * 20) - (avg_mae_1d * 0.01)))
        avg_conf_5d = min(94, max(83, 85 + (avg_r2_5d * 20) - (avg_mae_5d * 0.01)))
        avg_conf_30d = min(92, max(80, 85 + (avg_r2_30d * 20) - (avg_mae_30d * 0.01)))
        
        print(f"📈 Average Performance Metrics:")
        print(f"   1-day: MAE = ₹{avg_mae_1d:.2f}, R² = {avg_r2_1d:.4f}, Confidence = {avg_conf_1d:.1f}%")
        print(f"   5-day: MAE = ₹{avg_mae_5d:.2f}, R² = {avg_r2_5d:.4f}, Confidence = {avg_conf_5d:.1f}%")
        print(f"   30-day: MAE = ₹{avg_mae_30d:.2f}, R² = {avg_r2_30d:.4f}, Confidence = {avg_conf_30d:.1f}%")
    
    return training_results

def test_optimized_model(symbol: str):
    """Test a trained optimized model"""
    try:
        # Load data
        df = pd.read_csv(f"data/{symbol}_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create predictor and load trained model
        predictor = OptimizedPredictor()
        if predictor.load_optimized_model(symbol):
            predictions, confidence = predictor.predict_optimized(df)
            
            print(f"\n🔍 Optimized predictions for {symbol}:")
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
            print(f"❌ No optimized model found for {symbol}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error testing optimized model for {symbol}: {e}")
        return None, None

def compare_model_performance(symbols):
    """Compare performance across different models"""
    print(f"\n" + "=" * 70)
    print("🔄 Comparing Model Performance...")
    
    for symbol in symbols:
        print(f"\n📊 {symbol} Performance Comparison:")
        print("-" * 50)
        
        # Test optimized model
        try:
            df = pd.read_csv(f"data/{symbol}_data.csv")
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Optimized model
            opt_predictor = OptimizedPredictor()
            if opt_predictor.load_optimized_model(symbol):
                opt_pred, opt_conf = opt_predictor.predict_optimized(df)
                print(f"🚀 Optimized: 1d ₹{opt_pred['1_day']:.2f} ({opt_conf['1_day']:.1f}%), 5d ₹{opt_pred['5_day']:.2f} ({opt_conf['5_day']:.1f}%), 30d ₹{opt_pred['30_day']:.2f} ({opt_conf['30_day']:.1f}%)")
            else:
                print("❌ Optimized model not found")
            
            # Dynamic model
            try:
                from models.dynamic_predictor import DynamicPredictor
                dyn_predictor = DynamicPredictor()
                if dyn_predictor.load_model(symbol):
                    dyn_pred, dyn_conf = dyn_predictor.predict_dynamic(df)
                    print(f"⚡ Dynamic: 1d ₹{dyn_pred['1_day']:.2f} ({dyn_conf['1_day']:.1f}%), 5d ₹{dyn_pred['5_day']:.2f} ({dyn_conf['5_day']:.1f}%), 30d ₹{dyn_pred['30_day']:.2f} ({dyn_conf['30_day']:.1f}%)")
                else:
                    print("❌ Dynamic model not found")
            except:
                print("❌ Error with dynamic model")
            
        except Exception as e:
            print(f"❌ Error comparing models for {symbol}: {e}")

if __name__ == "__main__":
    # Train all optimized models
    results = train_all_optimized_models()
    
    # Test a few models
    print(f"\n" + "=" * 70)
    print("🧪 Testing Optimized Models...")
    
    # Get first few stocks for testing
    stocks = get_available_stocks()
    test_symbols = stocks[:3] if len(stocks) >= 3 else stocks
    
    for symbol in test_symbols:
        test_optimized_model(symbol)
        print("-" * 50)
    
    # Compare performance
    compare_model_performance(test_symbols)
