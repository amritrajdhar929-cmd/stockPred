import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedStockPredictor:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers
        self.models = {}
        self.feature_importance = {}
        self.is_trained = False
        self.validation_scores = {}
        
    def prepare_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for ML model"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Price-based features
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_3d'] = df['close'].pct_change(3)
        df['returns_7d'] = df['close'].pct_change(7)
        df['returns_30d'] = df['close'].pct_change(30)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Price relative to moving averages
        df['price_sma_5_ratio'] = df['close'] / df['sma_5']
        df['price_sma_20_ratio'] = df['close'] / df['sma_20']
        
        # Technical indicators
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['rsi_30'] = self.calculate_rsi(df['close'], 30)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].rolling(window=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_upper'] = df['sma_20'] + 2 * df['close'].rolling(window=20).std()
        df['bb_lower'] = df['sma_20'] - 2 * df['close'].rolling(window=20).std()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility features
        df['volatility_10d'] = df['close'].rolling(window=10).std()
        df['volatility_30d'] = df['close'].rolling(window=30).std()
        df['atr'] = self.calculate_atr(df)
        
        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        df['volume_price_trend'] = self.calculate_vpt(df)
        
        # Price patterns
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns_1d'].shift(lag)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with proper handling"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        price_change = df['close'].pct_change()
        vpt = (price_change * df['volume']).cumsum()
        return vpt
    
    def prepare_features_for_training(self, df: pd.DataFrame) -> tuple:
        """Prepare features and targets for training"""
        # Get all feature columns (exclude target and date)
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values
        feature_cols = [col for col in feature_cols if df[col].isna().sum() < len(df) * 0.3]
        
        # Prepare features
        X = df[feature_cols].copy()
        
        # Handle NaN values
        X = X.fillna(X.mean())
        
        # Prepare targets for different timeframes
        targets = {}
        targets['1_day'] = df['close'].shift(-1)
        targets['5_day'] = df['close'].shift(-5)
        targets['30_day'] = df['close'].shift(-30)
        
        return X, targets, feature_cols
    
    def train_with_validation(self, symbol: str, df: pd.DataFrame):
        """Train models with proper time series validation"""
        try:
            print(f"Training advanced model for {symbol}...")
            
            # Prepare features
            df_features = self.prepare_advanced_features(df)
            X, targets, feature_cols = self.prepare_features_for_training(df_features)
            
            # Remove rows with NaN targets
            valid_indices = []
            for timeframe in ['1_day', '5_day', '30_day']:
                valid_idx = targets[timeframe].notna()
                valid_indices.append(valid_idx)
            
            # Use intersection of all valid indices
            valid_mask = valid_indices[0]
            for mask in valid_indices[1:]:
                valid_mask = valid_mask & mask
            
            X_valid = X[valid_mask]
            targets_valid = {tf: targets[tf][valid_mask] for tf in targets}
            
            if len(X_valid) < 100:  # Need minimum data
                print(f"Insufficient data for {symbol}: {len(X_valid)} samples")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_valid)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train models for each timeframe
            for timeframe in ['1_day', '5_day', '30_day']:
                print(f"Training {timeframe} model for {symbol}...")
                
                y = targets_valid[timeframe].values
                
                # XGBoost model with optimized parameters
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                         scoring='neg_mean_absolute_error')
                
                # Train on full dataset
                model.fit(X_scaled, y)
                
                # Store model and validation scores
                self.models[timeframe] = model
                self.validation_scores[timeframe] = {
                    'mae': -cv_scores.mean(),
                    'mae_std': cv_scores.std(),
                    'samples': len(X_valid)
                }
                
                # Feature importance
                self.feature_importance[timeframe] = dict(zip(
                    feature_cols, model.feature_importances_
                ))
                
                print(f"{timeframe} MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
            
            self.is_trained = True
            self.feature_cols = feature_cols
            
            # Save model
            self.save_model(symbol)
            
            print(f"✓ Successfully trained advanced model for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error training advanced model for {symbol}: {e}")
            self.is_trained = False
            return False
    
    def predict_with_confidence(self, df: pd.DataFrame) -> tuple:
        """Make predictions with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            # Prepare features
            df_features = self.prepare_advanced_features(df)
            X, _, _ = self.prepare_features_for_training(df_features)
            
            # Handle NaN values
            X = X.fillna(X.mean())
            
            # Get the latest data point
            X_latest = X.iloc[-1:].values
            X_scaled = self.scaler.transform(X_latest)
            
            predictions = {}
            confidence_scores = {}
            
            for timeframe in ['1_day', '5_day', '30_day']:
                if timeframe in self.models:
                    model = self.models[timeframe]
                    
                    # Get prediction
                    pred = model.predict(X_scaled)[0]
                    predictions[timeframe] = float(pred)
                    
                    # Calculate confidence based on validation performance
                    val_score = self.validation_scores[timeframe]
                    base_confidence = max(50, min(95, 100 - (val_score['mae'] / df['close'].iloc[-1]) * 1000))
                    
                    # Adjust confidence based on prediction uncertainty
                    if hasattr(model, 'predict'):
                        # For XGBoost, use standard deviation of trees if available
                        try:
                            tree_preds = []
                            for estimator in model.estimators_:
                                tree_pred = estimator.predict(X_scaled)[0]
                                tree_preds.append(tree_pred)
                            
                            if tree_preds:
                                pred_std = np.std(tree_preds)
                                uncertainty_penalty = min(20, pred_std / df['close'].iloc[-1] * 100)
                                confidence = base_confidence - uncertainty_penalty
                            else:
                                confidence = base_confidence
                        except:
                            confidence = base_confidence
                    else:
                        confidence = base_confidence
                    
                    confidence_scores[timeframe] = max(50, min(95, confidence))
            
            return predictions, confidence_scores
            
        except Exception as e:
            print(f"Error in advanced prediction: {e}")
            raise
    
    def save_model(self, symbol: str):
        """Save trained model"""
        try:
            model_dir = "backend/saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'feature_importance': self.feature_importance,
                'validation_scores': self.validation_scores,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, f"{model_dir}/{symbol}_advanced_model.pkl")
            
        except Exception as e:
            print(f"Error saving advanced model: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load trained model"""
        try:
            model_path = f"backend/saved_models/{symbol}_advanced_model.pkl"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.feature_cols = model_data['feature_cols']
                self.feature_importance = model_data.get('feature_importance', {})
                self.validation_scores = model_data.get('validation_scores', {})
                self.is_trained = model_data['is_trained']
                return True
        except Exception as e:
            print(f"Error loading advanced model: {e}")
        
        return False
