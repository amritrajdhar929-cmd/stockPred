import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ConservativePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
        self.validation_scores = {}
        self.feature_cols = []
        
    def generate_realistic_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Generate realistic synthetic stock data with proper patterns"""
        np.random.seed(42)
        
        # Base price based on stock symbol (realistic ranges)
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
        
        # Generate realistic price movements
        # Use geometric Brownian motion with mean reversion
        dt = 1/252  # Daily time step
        mu = 0.08  # Annual drift (8% return)
        sigma = 0.20  # Annual volatility (20%)
        theta = 0.1  # Mean reversion speed
        
        prices = [base_price]
        for i in range(1, days):
            # Geometric Brownian motion with mean reversion
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
            
            # Mean reversion force
            mean_reversion = -theta * (np.log(prices[-1]/base_price)) * dt
            
            new_price = prices[-1] * np.exp(drift + diffusion + mean_reversion)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Generate OHLC data
        opens = prices * (1 + np.random.normal(0, 0.005, days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, days)))
        
        # Generate realistic volume
        base_volume = np.random.randint(1000000, 5000000)
        volumes = base_volume * (1 + np.random.normal(0, 0.3, days))
        volumes = np.maximum(volumes, 100000)  # Minimum volume
        
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
    
    def prepare_conservative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare conservative features with financial logic"""
        df = df.copy()
        df = df.sort_values('date')
        
        # Basic price features
        df['returns_1d'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Price relative to moving averages
        df['price_sma5_ratio'] = df['close'] / df['sma_5']
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['price_sma50_ratio'] = df['close'] / df['sma_50']
        
        # RSI
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        
        # Volatility
        df['volatility_20d'] = df['close'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_20d'] / df['close'].rolling(20).mean()
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price momentum
        df['momentum_5'] = (df['close'] / df['close'].shift(5)) - 1
        df['momentum_20'] = (df['close'] / df['close'].shift(20)) - 1
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_conservative_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create conservative target variables"""
        df_targets = df.copy()
        
        # Calculate future returns with very conservative bounds
        df_targets['future_return_1d'] = df_targets['close'].shift(-1) / df_targets['close'] - 1
        df_targets['future_return_5d'] = df_targets['close'].shift(-5) / df_targets['close'] - 1
        df_targets['future_return_30d'] = df_targets['close'].shift(-30) / df_targets['close'] - 1
        
        # Very conservative bounds
        df_targets['future_return_1d'] = df_targets['future_return_1d'].clip(-0.02, 0.02)  # ±2% daily
        df_targets['future_return_5d'] = df_targets['future_return_5d'].clip(-0.05, 0.05)  # ±5% weekly
        df_targets['future_return_30d'] = df_targets['future_return_30d'].clip(-0.10, 0.10)  # ±10% monthly
        
        # Convert to price targets
        df_targets['target_1d'] = df_targets['close'] * (1 + df_targets['future_return_1d'])
        df_targets['target_5d'] = df_targets['close'] * (1 + df_targets['future_return_5d'])
        df_targets['target_30d'] = df_targets['close'] * (1 + df_targets['future_return_30d'])
        
        return df_targets
    
    def train_conservative_model(self, symbol: str, df: pd.DataFrame = None):
        """Train conservative model"""
        try:
            print(f"🚀 Training conservative model for {symbol}...")
            
            # Generate realistic data if not provided
            if df is None:
                df = self.generate_realistic_data(symbol)
            
            # Prepare features and targets
            df_features = self.prepare_conservative_features(df)
            df_targets = self.create_conservative_targets(df_features)
            
            # Define feature columns
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 
                          'future_return_1d', 'future_return_5d', 'future_return_30d',
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
                
                if len(X_valid) < 100:
                    print(f"❌ Insufficient data for {tf}: {len(X_valid)} samples")
                    continue
                
                # Create conservative model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Simple train/test split (80/20)
                split_idx = int(len(X_valid) * 0.8)
                X_train, X_test = X_valid[:split_idx], X_valid[split_idx:]
                y_train, y_test = y_valid[:split_idx], y_valid[split_idx:]
                
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
            
            print(f"✅ Conservative model trained for {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error training conservative model for {symbol}: {e}")
            self.is_trained = False
            return False
    
    def predict_conservative(self, df: pd.DataFrame, current_price: float) -> tuple:
        """Make conservative predictions"""
        if not self.is_trained:
            raise ValueError("Conservative model not trained yet")
        
        try:
            # Prepare features
            df_features = self.prepare_conservative_features(df)
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
                    
                    # Apply very conservative bounds
                    if tf == '1d':
                        # Daily: ±2% max
                        pred = max(current_price * 0.98, min(current_price * 1.02, pred))
                        base_confidence = 80
                    elif tf == '5d':
                        # Weekly: ±5% max
                        pred = max(current_price * 0.95, min(current_price * 1.05, pred))
                        base_confidence = 75
                    else:  # 30d
                        # Monthly: ±10% max
                        pred = max(current_price * 0.90, min(current_price * 1.10, pred))
                        base_confidence = 70
                    
                    # Calculate confidence based on validation performance
                    val_score = self.validation_scores[tf]
                    mae_ratio = val_score['mae'] / current_price
                    confidence_adjustment = max(-5, min(5, (1 - mae_ratio * 100) * 5))
                    
                    final_confidence = base_confidence + confidence_adjustment
                    final_confidence = max(65, min(85, final_confidence))
                    
                    # Store results
                    predictions[f'{tf}_day'] = float(pred)
                    confidence_scores[f'{tf}_day'] = float(final_confidence)
            
            return predictions, confidence_scores
            
        except Exception as e:
            print(f"❌ Error in conservative prediction: {e}")
            raise
    
    def save_model(self, symbol: str):
        """Save conservative model"""
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
            
            joblib.dump(model_data, f"{model_dir}/{symbol}_conservative_model.pkl")
            
        except Exception as e:
            print(f"❌ Error saving conservative model: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load conservative model"""
        try:
            model_path = f"backend/saved_models/{symbol}_conservative_model.pkl"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.feature_cols = model_data['feature_cols']
                self.validation_scores = model_data['validation_scores']
                self.is_trained = model_data['is_trained']
                return True
        except Exception as e:
            print(f"❌ Error loading conservative model: {e}")
        
        return False
