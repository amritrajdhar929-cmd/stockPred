#!/usr/bin/env python3
"""
Realistic Stock Prediction Model Training Script V2
Improved model with better accuracy and realistic predictions
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.realistic_predictor_v2 import RealisticPredictorV2

# NSE Stock List
NSE_STOCKS = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd."},
    {"symbol": "TCS", "name": "Tata Consultancy Services"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd."},
    {"symbol": "INFY", "name": "Infosys Ltd."},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd."},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd."},
    {"symbol": "SBIN", "name": "State Bank of India"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd."},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd."},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Ltd."},
    {"symbol": "WIPRO", "name": "Wipro Ltd."},
    {"symbol": "AXISBANK", "name": "Axis Bank Ltd."},
    {"symbol": "ITC", "name": "ITC Ltd."},
    {"symbol": "HCLTECH", "name": "HCL Technologies Ltd."},
    {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd."},
    {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd."},
    {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd."},
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries Ltd."},
    {"symbol": "TITAN", "name": "Titan Company Ltd."},
    {"symbol": "NTPC", "name": "NTPC Ltd."},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Ltd."},
    {"symbol": "TECHM", "name": "Tech Mahindra Ltd."},
    {"symbol": "GRASIM", "name": "Grasim Industries Ltd."},
    {"symbol": "POWERGRID", "name": "Power Grid Corporation of India"},
    {"symbol": "COALINDIA", "name": "Coal India Ltd."},
    {"symbol": "BPCL", "name": "Bharat Petroleum Corporation Ltd."},
    {"symbol": "ONGC", "name": "Oil and Natural Gas Corporation Ltd."},
    {"symbol": "JSWSTEEL", "name": "JSW Steel Ltd."},
    {"symbol": "DIVISLAB", "name": "Divi's Laboratories Ltd."},
    {"symbol": "DRREDDY", "name": "Dr. Reddy's Laboratories Ltd."}
]

def get_nse_stock_price(symbol: str) -> float:
    """Get current stock price (simulated)"""
    # Simulate current prices based on realistic market data
    base_prices = {
        'RELIANCE': 2400, 'TCS': 3500, 'INFY': 1500, 'HDFCBANK': 1600,
        'ICICIBANK': 900, 'SBIN': 600, 'ASIANPAINT': 2400, 'BAJFINANCE': 7000,
        'HINDUNILVR': 2500, 'ITC': 400, 'MARUTI': 10000, 'KOTAKBANK': 1800,
        'WIPRO': 400, 'TITAN': 2500, 'ULTRACEMCO': 8000, 'JSWSTEEL': 800,
        'NTPC': 200, 'ONGC': 150, 'POWERGRID': 200, 'GRASIM': 1800,
        'TECHM': 1200, 'HCLTECH': 1300, 'SUNPHARMA': 900, 'TATAMOTORS': 400,
        'AXISBANK': 1000, 'M&M': 1400, 'BPCL': 400, 'COALINDIA': 200
    }
    
    # Add some random variation to simulate market movement
    base_price = base_prices.get(symbol, 1000)
    variation = np.random.normal(0, 0.02)  # ±2% variation
    return round(base_price * (1 + variation), 2)

def main():
    """Main training function"""
    print("🚀 Starting Realistic Model Training V2...")
    print("=" * 80)
    
    # Initialize predictor
    predictor = RealisticPredictorV2()
    
    # Training results
    results = {}
    successful_trainings = 0
    failed_trainings = 0
    
    # Train models for all stocks
    for i, stock in enumerate(NSE_STOCKS, 1):
        symbol = stock['symbol']
        print(f"\n[{i:2d}/{len(NSE_STOCKS)}] 🎯 Training realistic model for {symbol}...")
        
        try:
            # Generate realistic data
            df = predictor.generate_realistic_data(symbol, days=365)
            
            # Train model
            if predictor.train_model(symbol, df):
                # Test prediction
                current_price = get_nse_stock_price(symbol)
                predictions, confidence_scores = predictor.predict(df, current_price)
                
                # Store results
                results[symbol] = {
                    'current_price': current_price,
                    'predictions': predictions,
                    'confidence_scores': confidence_scores,
                    'validation_scores': predictor.validation_scores
                }
                
                successful_trainings += 1
                print(f"✅ Realistic model trained for {symbol}")
                
                # Display sample predictions
                print(f"📊 Model Performance:")
                for tf in ['1d', '5d', '30d']:
                    if tf in predictor.validation_scores:
                        val = predictor.validation_scores[tf]
                        print(f"   {tf}: MAE = ₹{val['mae']:.2f}, R² = {val['r2']:.4f}")
                
                # Display sample predictions
                print(f"🔍 Sample predictions for {symbol}:")
                print(f"💰 Current price: ₹{current_price:.2f}")
                for tf in ['1d', '5d', '30d']:
                    if f'{tf}_day' in predictions:
                        pred = predictions[f'{tf}_day']
                        conf = confidence_scores[f'{tf}_day']
                        print(f"📈 {tf}: ₹{pred:.2f} (confidence: {conf:.1f}%)")
                print("-" * 50)
            else:
                failed_trainings += 1
                print(f"❌ Failed to train realistic model for {symbol}")
                
        except Exception as e:
            failed_trainings += 1
            print(f"❌ Error training realistic model for {symbol}: {e}")
    
    # Save training results
    results_file = "backend/realistic_training_results_v2.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 Results saved to {results_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 Realistic Training Summary:")
    print(f"✅ Successfully trained: {successful_trainings}/{len(NSE_STOCKS)} models")
    print(f"❌ Failed: {failed_trainings}/{len(NSE_STOCKS)} models")
    print(f"📁 Models saved in: backend/saved_models")
    print(f"📊 Results saved in: {results_file}")
    
    # Calculate average performance
    if results:
        avg_mae_1d = np.mean([r['validation_scores']['1d']['mae'] for r in results.values() if '1d' in r['validation_scores']])
        avg_r2_1d = np.mean([r['validation_scores']['1d']['r2'] for r in results.values() if '1d' in r['validation_scores']])
        
        print(f"\n📈 Average Performance (1-day):")
        print(f"   MAE = ₹{avg_mae_1d:.2f}, R² = {avg_r2_1d:.4f}")
    
    print("\n🎉 Realistic Model Training Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
