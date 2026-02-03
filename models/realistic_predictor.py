import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealisticPredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.models = {}
        self.is_trained = False
        self.validation_scores = {}
        self.feature_cols = []
        
    def prepare_realistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare realistic features with proper financial logic"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
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
        
        # Open-Close spread
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
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
    
    def create_realistic_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create realistic target variables based on price movements"""
        df_targets = df.copy()
        
        # Calculate future returns with realistic bounds
        df_targets['future_return_1d'] = df_targets['close'].shift(-1) / df_targets['close'] - 1
        df_targets['future_return_5d'] = df_targets['close'].shift(-5) / df_targets['close'] - 1
        df_targets['future_return_30d'] = df_targets['close'].shift(-30) / df_targets['close'] - 1
        
        # Clip extreme returns to realistic values
        df_targets['future_return_1d'] = df_targets['future_return_1d'].clip(-0.05, 0.05)  # ±5% daily
        df_targets['future_return_5d'] = df_targets['future_return_5d'].clip(-0.15, 0.15)  # ±15% weekly
        df_targets['future_return_30d'] = df_targets['future_return_30d'].clip(-0.30, 0.30)  # ±30% monthly
        
        # Convert to price targets
        df_targets['target_1d'] = df_targets['close'] * (1 + df_targets['future_return_1d'])
        df_targets['target_5d'] = df_targets['close'] * (1 + df_targets['future_return_5d'])
        df_targets['target_30d'] = df_targets['close'] * (1 + df_targets['future_return_30d'])
        
        return df_targets
    
    def train_realistic_model(self, symbol: str, df: pd.DataFrame):
        """Train realistic model with proper validation"""
        try:
            print(f"🚀 Training realistic model for {symbol}...")
            
            # Prepare features and targets
            df_features = self.prepare_realistic_features(df)
            df_targets = self.create_realistic_targets(df_features)
            
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
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)
            
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
                
                # Create model with conservative parameters
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Cross-validation
                cv_mae = cross_val_score(model, X_valid, y_valid, cv=tscv, 
                                       scoring='neg_mean_absolute_error')
                cv_r2 = cross_val_score(model, X_valid, y_valid, cv=tscv, 
                                      scoring='r2')
                
                # Train on full dataset
                model.fit(X_valid, y_valid)
                
                # Store model and scores
                self.models[tf] = model
                self.validation_scores[tf] = {
                    'mae': -cv_mae.mean(),
                    'mae_std': cv_mae.std(),
                    'r2': cv_r2.mean(),
                    'r2_std': cv_r2.std(),
                    'samples': len(X_valid)
                }
                
                print(f"   ✅ {tf}: MAE = {-cv_mae.mean():.2f}, R² = {cv_r2.mean():.4f}")
            
            self.is_trained = True
            self.save_model(symbol)
            
            print(f"✅ Realistic model trained for {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error training realistic model for {symbol}: {e}")
            self.is_trained = False
            return False
    
    def predict_realistic(self, df: pd.DataFrame, current_price: float) -> tuple:
        """Make realistic predictions with proper bounds"""
        if not self.is_trained:
            raise ValueError("Realistic model not trained yet")
        
        try:
            # Prepare features
            df_features = self.prepare_realistic_features(df)
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
                    
                    # Apply realistic bounds based on timeframe
                    if tf == '1d':
                        # Daily: ±3% max
                        pred = max(current_price * 0.97, min(current_price * 1.03, pred))
                        base_confidence = 85
                    elif tf == '5d':
                        # Weekly: ±10% max
                        pred = max(current_price * 0.90, min(current_price * 1.10, pred))
                        base_confidence = 80
                    else:  # 30d
                        # Monthly: ±20% max
                        pred = max(current_price * 0.80, min(current_price * 1.20, pred))
                        base_confidence = 75
                    
                    # Calculate confidence based on validation performance
                    val_score = self.validation_scores[tf]
                    mae_ratio = val_score['mae'] / current_price
                    confidence_adjustment = max(-10, min(10, (1 - mae_ratio * 100) * 10))
                    
                    final_confidence = base_confidence + confidence_adjustment
                    final_confidence = max(70, min(95, final_confidence))
                    
                    # Store results
                    predictions[f'{tf}_day'] = float(pred)
                    confidence_scores[f'{tf}_day'] = float(final_confidence)
            
            return predictions, confidence_scores
            
        except Exception as e:
            print(f"❌ Error in realistic prediction: {e}")
            raise
    
    def save_model(self, symbol: str):
        """Save realistic model"""
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
            
            joblib.dump(model_data, f"{model_dir}/{symbol}_realistic_model.pkl")
            
        except Exception as e:
            print(f"❌ Error saving realistic model: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load realistic model"""
        try:
            model_path = f"backend/saved_models/{symbol}_realistic_model.pkl"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.feature_cols = model_data['feature_cols']
                self.validation_scores = model_data['validation_scores']
                self.is_trained = model_data['is_trained']
                return True
        except Exception as e:
            print(f"❌ Error loading realistic model: {e}")
        
        return False
