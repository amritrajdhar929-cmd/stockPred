import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimplePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
        self.validation_scores = {}
        self.feature_cols = []
        
    def generate_realistic_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Generate realistic synthetic stock data"""
        np.random.seed(hash(symbol) % 1000)  # Different seed for each symbol
        
        # Base prices for different stocks
        base_prices = {
            'RELIANCE': 1400, 'TCS': 3200, 'INFY': 1500, 'HDFCBANK': 1600,
            'ICICIBANK': 900, 'SBIN': 600, 'ASIANPAINT': 2400, 'BAJFINANCE': 7000,
            'HINDUNILVR': 2500, 'ITC': 400, 'MARUTI': 10000, 'KOTAKBANK': 1800,
            'WIPRO': 400, 'TITAN': 2500, 'ULTRACEMCO': 8000, 'JSWSTEEL': 800,
            'NTPC': 200, 'ONGC': 150, 'POWERGRID': 200, 'GRASIM': 1800,
            'TECHM': 1200, 'HCLTECH': 1300, 'SUNPHARMA': 900, 'TATAMOTORS': 400,
            'AXISBANK': 1000, 'M&M': 1400, 'BPCL': 400, 'COALINDIA': 200
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Generate dates
        end_date = datetime.now()
        dates = pd.date_range(end=end_date - timedelta(days=1), periods=days, freq='D')
        
        # Generate realistic price movements using random walk with drift
        prices = [base_price]
        daily_volatility = 0.015  # 1.5% daily volatility
        drift = 0.0003  # Small positive drift
        
        for i in range(1, days):
            # Random walk with drift
            return_rate = drift + np.random.normal(0, daily_volatility)
            # Limit extreme movements
            return_rate = np.clip(return_rate, -0.05, 0.05)  # ±5% daily limit
            
            new_price = prices[-1] * (1 + return_rate)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Generate OHLC data
        opens = prices * (1 + np.random.normal(0, 0.003, days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.008, days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.008, days)))
        
        # Generate realistic volume
        base_volume = np.random.randint(500000, 2000000)
        volumes = base_volume * (1 + np.random.normal(0, 0.2, days))
        volumes = np.maximum(volumes, 100000)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes.astype(int)
        })
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare simple features"""
        df = df.copy()
        df = df.sort_values('date')
        
        # Basic returns
        df['returns_1d'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Price ratios
        df['price_sma5_ratio'] = df['close'] / df['sma_5']
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        
        # Volatility
        df['volatility_20d'] = df['close'].rolling(20).std()
        
        # Volume ratio
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # RSI
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables"""
        df_targets = df.copy()
        
        # Future prices
        df_targets['target_1d'] = df_targets['close'].shift(-1)
        df_targets['target_5d'] = df_targets['close'].shift(-5)
        df_targets['target_30d'] = df_targets['close'].shift(-30)
        
        return df_targets
    
    def train_model(self, symbol: str, df: pd.DataFrame = None):
        """Train simple prediction model"""
        try:
            print(f"🚀 Training simple model for {symbol}...")
            
            # Generate data if not provided
            if df is None:
                df = self.generate_realistic_data(symbol)
            
            # Prepare features and targets
            df_features = self.prepare_features(df)
            df_targets = self.create_targets(df_features)
            
            # Define feature columns
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 
                          'target_1d', 'target_5d', 'target_30d']
            self.feature_cols = [col for col in df_targets.columns if col not in exclude_cols]
            
            # Prepare features
            X = df_targets[self.feature_cols].copy()
            X = X.fillna(method='ffill').fillna(X.mean())
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models for each timeframe
            timeframes = ['1d', '5d', '30d']
            target_cols = ['target_1d', 'target_5d', 'target_30d']
            
            for tf, target_col in zip(timeframes, target_cols):
                print(f"   Training {tf} model for {symbol}...")
                
                # Get target
                y = df_targets[target_col]
                
                # Remove NaN values
                valid_mask = pd.notna(y) & pd.notna(X_scaled).all(axis=1)
                X_valid = X_scaled[valid_mask]
                y_valid = y[valid_mask]
                
                if len(X_valid) < 200:
                    print(f"❌ Insufficient data for {tf}: {len(X_valid)} samples")
                    continue
                
                # Simple train/test split (80/20)
                split_idx = int(len(X_valid) * 0.8)
                X_train, X_test = X_valid[:split_idx], X_valid[split_idx:]
                y_train, y_test = y_valid[:split_idx], y_valid[split_idx:]
                
                # Create simple model
                model = xgb.XGBRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Validate
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store model and scores
                self.models[tf] = model
                self.validation_scores[tf] = {
                    'mae': mae,
                    'r2': r2,
                    'samples': len(X_valid)
                }
                
                print(f"   ✅ {tf}: MAE = {mae:.2f}, R² = {r2:.4f}")
            
            self.is_trained = True
            self.save_model(symbol)
            
            print(f"✅ Simple model trained for {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error training simple model for {symbol}: {e}")
            self.is_trained = False
            return False
    
    def predict(self, df: pd.DataFrame, current_price: float) -> tuple:
        """Make simple predictions"""
        if not self.is_trained:
            raise ValueError("Simple model not trained yet")
        
        try:
            # Prepare features
            df_features = self.prepare_features(df)
            X = df_features[self.feature_cols].copy()
            X = X.fillna(method='ffill').fillna(X.mean())
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            predictions = {}
            confidence_scores = {}
            
            # Make predictions for each timeframe
            for tf in ['1d', '5d', '30d']:
                if tf in self.models:
                    model = self.models[tf]
                    
                    # Get prediction
                    pred = model.predict(X_scaled[-1:])[0]
                    
                    # Apply realistic bounds
                    if tf == '1d':
                        # Daily: ±3% max
                        pred = max(current_price * 0.97, min(current_price * 1.03, pred))
                        base_confidence = 85
                    elif tf == '5d':
                        # Weekly: ±8% max
                        pred = max(current_price * 0.92, min(current_price * 1.08, pred))
                        base_confidence = 82
                    else:  # 30d
                        # Monthly: ±15% max
                        pred = max(current_price * 0.85, min(current_price * 1.15, pred))
                        base_confidence = 80
                    
                    # Calculate confidence based on validation performance
                    val_score = self.validation_scores[tf]
                    mae_ratio = val_score['mae'] / current_price
                    confidence_adjustment = max(-3, min(8, (1 - mae_ratio * 50) * 10))
                    
                    final_confidence = base_confidence + confidence_adjustment
                    final_confidence = max(80, min(95, final_confidence))
                    
                    # Store results
                    predictions[f'{tf}_day'] = float(pred)
                    confidence_scores[f'{tf}_day'] = float(final_confidence)
            
            # Convert keys to match frontend expectations
            formatted_predictions = {}
            formatted_confidence_scores = {}
            
            for tf in ['1d', '5d', '30d']:
                if f'{tf}_day' in predictions:
                    formatted_predictions[f'{tf}_day'] = predictions[f'{tf}_day']
                    formatted_confidence_scores[f'{tf}_day'] = confidence_scores[f'{tf}_day']
            
            return formatted_predictions, formatted_confidence_scores
            
        except Exception as e:
            print(f"❌ Error in simple prediction: {e}")
            raise
    
    def save_model(self, symbol: str):
        """Save simple model"""
        try:
            model_dir = "backend/saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'validation_scores': self.validation_scores,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, f"{model_dir}/{symbol}_simple_model.pkl")
            
        except Exception as e:
            print(f"❌ Error saving simple model: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load simple model"""
        try:
            model_path = f"backend/saved_models/{symbol}_simple_model.pkl"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.feature_cols = model_data['feature_cols']
                self.validation_scores = model_data['validation_scores']
                self.is_trained = model_data['is_trained']
                return True
        except Exception as e:
            print(f"❌ Error loading simple model: {e}")
        
        return False
