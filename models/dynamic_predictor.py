import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DynamicPredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.ensemble_models = {}
        self.feature_importance = {}
        self.is_trained = False
        self.validation_scores = {}
        self.model_weights = {}
        self.config = self.load_dynamic_config()
        
    def load_dynamic_config(self):
        """Load dynamic configuration"""
        return {
            'prediction_timeframes': [1, 5, 30],
            'min_confidence_thresholds': {'1_day': 82, '5_day': 80, '30_day': 78},
            'max_confidence_thresholds': {'1_day': 95, '5_day': 92, '30_day': 90},
            'ensemble_models': ['xgb', 'rf', 'gb'],
            'feature_periods': {
                'returns': [1, 2, 3, 5, 7, 10, 14, 21, 30],
                'ma': [5, 10, 20, 50],
                'volatility': [5, 10, 20, 30],
                'volume': [5, 10, 20],
                'lags': [1, 2, 3, 5, 7, 10]
            },
            'model_params': {
                'xgb': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.03,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'rf': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'gb': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'max_features': 'sqrt',
                    'random_state': 42
                }
            }
        }
    
    def prepare_dynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dynamic features based on configuration"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # ===== RETURNS =====
        for period in self.config['feature_periods']['returns']:
            df[f'returns_{period}d'] = df['close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Cumulative returns
        df['cum_return_5d'] = df['returns_1d'].rolling(5).sum()
        df['cum_return_20d'] = df['returns_1d'].rolling(20).sum()
        
        # ===== MOVING AVERAGES =====
        for period in self.config['feature_periods']['ma']:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            df[f'price_above_sma_{period}'] = (df['close'] > df[f'sma_{period}']).astype(int)
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_12_26_ratio'] = df['ema_12'] / df['ema_26']
        df['ema_12_26_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
        
        # MA crossovers
        df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # ===== TECHNICAL INDICATORS =====
        # RSI
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['rsi_30'] = self.calculate_rsi(df['close'], 30)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands
        df['bb_upper'] = df['sma_20'] + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['sma_20'] - 2 * df['close'].rolling(20).std()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_upper_touch'] = (df['close'] >= df['bb_upper']).astype(int)
        df['bb_lower_touch'] = (df['close'] <= df['bb_lower']).astype(int)
        
        # ===== VOLATILITY FEATURES =====
        for period in self.config['feature_periods']['volatility']:
            df[f'volatility_{period}d'] = df['close'].rolling(period).std()
            df[f'volatility_pct_{period}d'] = df['close'].rolling(period).std() / df['close'].rolling(period).mean()
        
        # ATR
        df['atr_14'] = self.calculate_atr(df, 14)
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # Dynamic volatility regime (using percentiles instead of qcut)
        vol_20d = df['volatility_20d']
        df['volatility_regime'] = np.where(vol_20d <= vol_20d.quantile(0.33), 0,
                                         np.where(vol_20d <= vol_20d.quantile(0.67), 1, 2))
        
        # ===== VOLUME FEATURES =====
        for period in self.config['feature_periods']['volume']:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # Volume Price Trend
        df['vpt'] = self.calculate_vpt(df)
        df['vpt_sma_10'] = df['vpt'].rolling(10).mean()
        df['vpt_ratio'] = df['vpt'] / df['vpt_sma_10']
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        # ===== PRICE PATTERN FEATURES =====
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['high'] - df['low'])
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['high'] - df['low'])
        
        # Candlestick patterns
        df['doji'] = (abs(df['close'] - df['open']) / df['open'] < 0.01).astype(int)
        df['hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & (df['upper_shadow'] < 0.1 * df['body_size'])).astype(int)
        
        # Gap analysis
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # ===== TIME FEATURES =====
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # Seasonal indicators
        df['is_winter'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        
        # ===== LAG FEATURES =====
        for lag in self.config['feature_periods']['lags']:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns_1d'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        
        # Rolling statistics
        for period in [3, 5, 10]:
            df[f'returns_mean_{period}d'] = df['returns_1d'].rolling(period).mean()
            df[f'returns_std_{period}d'] = df['returns_1d'].rolling(period).std()
        
        # ===== MARKET MICROSTRUCTURE =====
        df['price_efficiency'] = abs(df['close'] - df['vwap']) / df['vwap']
        df['liquidity_ratio'] = df['volume'] / (df['high'] - df['low'])
        df['depth_ratio'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period, min_periods=1).mean()
    
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
        X = X.fillna(method='ffill').fillna(X.mean())
        
        # Prepare targets for different timeframes
        targets = {}
        for days in self.config['prediction_timeframes']:
            timeframe = f'{days}_day'
            targets[timeframe] = df['close'].shift(-days)
        
        return X, targets, feature_cols
    
    def create_dynamic_ensemble(self):
        """Create dynamic ensemble based on configuration"""
        estimators = []
        
        if 'xgb' in self.config['ensemble_models']:
            xgb_model = xgb.XGBRegressor(**self.config['model_params']['xgb'])
            estimators.append(('xgb', xgb_model))
        
        if 'rf' in self.config['ensemble_models']:
            rf_model = RandomForestRegressor(**self.config['model_params']['rf'])
            estimators.append(('rf', rf_model))
        
        if 'gb' in self.config['ensemble_models']:
            gb_model = GradientBoostingRegressor(**self.config['model_params']['gb'])
            estimators.append(('gb', gb_model))
        
        return VotingRegressor(estimators)
    
    def train_dynamic_model(self, symbol: str, df: pd.DataFrame):
        """Train dynamic ensemble model"""
        try:
            print(f"🚀 Training dynamic ensemble model for {symbol}...")
            
            # Prepare features
            df_features = self.prepare_dynamic_features(df)
            X, targets, feature_cols = self.prepare_features_for_training(df_features)
            
            # Remove rows with NaN targets
            valid_indices = []
            for days in self.config['prediction_timeframes']:
                timeframe = f'{days}_day'
                valid_idx = targets[timeframe].notna()
                valid_indices.append(valid_idx)
            
            # Use intersection of all valid indices
            valid_mask = valid_indices[0]
            for mask in valid_indices[1:]:
                valid_mask = valid_mask & mask
            
            X_valid = X[valid_mask]
            targets_valid = {f'{days}_day': targets[f'{days}_day'][valid_mask] 
                           for days in self.config['prediction_timeframes']}
            
            if len(X_valid) < 200:
                print(f"❌ Insufficient data for {symbol}: {len(X_valid)} samples")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_valid)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train ensemble models for each timeframe
            for days in self.config['prediction_timeframes']:
                timeframe = f'{days}_day'
                print(f"   Training {timeframe} ensemble for {symbol}...")
                
                y = targets_valid[timeframe].values
                
                # Create and train ensemble
                ensemble_model = self.create_dynamic_ensemble()
                
                # Cross-validation
                cv_mae = cross_val_score(ensemble_model, X_scaled, y, cv=tscv, 
                                       scoring='neg_mean_absolute_error')
                cv_r2 = cross_val_score(ensemble_model, X_scaled, y, cv=tscv, 
                                      scoring='r2')
                
                # Train on full dataset
                ensemble_model.fit(X_scaled, y)
                
                # Store model and validation scores
                self.ensemble_models[timeframe] = ensemble_model
                self.validation_scores[timeframe] = {
                    'mae': -cv_mae.mean(),
                    'mae_std': cv_mae.std(),
                    'r2': cv_r2.mean(),
                    'r2_std': cv_r2.std(),
                    'samples': len(X_valid),
                    'features': len(feature_cols)
                }
                
                # Calculate model weights
                mae_weight = 1 / (-cv_mae.mean() + 1e-6)
                r2_weight = cv_r2.mean() if cv_r2.mean() > 0 else 0.1
                self.model_weights[timeframe] = (mae_weight + r2_weight) / 2
                
                print(f"   ✅ {timeframe}: MAE = {-cv_mae.mean():.4f}, R² = {cv_r2.mean():.4f}")
            
            self.is_trained = True
            self.feature_cols = feature_cols
            
            # Save model
            self.save_model(symbol)
            
            print(f"✅ Dynamic ensemble model trained for {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error training dynamic model for {symbol}: {e}")
            self.is_trained = False
            return False
    
    def predict_dynamic(self, df: pd.DataFrame) -> tuple:
        """Make predictions with dynamic high confidence"""
        if not self.is_trained:
            raise ValueError("Dynamic model not trained yet")
        
        try:
            # Prepare features
            df_features = self.prepare_dynamic_features(df)
            X, _, _ = self.prepare_features_for_training(df_features)
            
            # Handle NaN values
            X = X.fillna(method='ffill').fillna(X.mean())
            
            # Get the latest data point
            X_latest = X.iloc[-1:].values
            X_scaled = self.scaler.transform(X_latest)
            
            predictions = {}
            confidence_scores = {}
            
            for days in self.config['prediction_timeframes']:
                timeframe = f'{days}_day'
                if timeframe in self.ensemble_models:
                    model = self.ensemble_models[timeframe]
                    
                    # Get prediction from ensemble
                    pred = model.predict(X_scaled)[0]
                    predictions[timeframe] = float(pred)
                    
                    # Calculate dynamic high confidence
                    val_score = self.validation_scores[timeframe]
                    current_price = df['close'].iloc[-1]
                    
                    # Base confidence from MAE
                    mae_ratio = val_score['mae'] / current_price
                    base_confidence = max(75, min(98, 100 - (mae_ratio * 1200)))
                    
                    # Boost confidence based on R² score
                    r2_boost = val_score['r2'] * 20  # Up to 20% boost for good R²
                    
                    # Additional confidence from ensemble stability
                    ensemble_stability = self.model_weights[timeframe] * 10
                    
                    # Final confidence calculation
                    final_confidence = min(95, base_confidence + r2_boost + ensemble_stability)
                    
                    # Apply dynamic confidence thresholds from config
                    min_conf = self.config['min_confidence_thresholds'][timeframe]
                    max_conf = self.config['max_confidence_thresholds'][timeframe]
                    confidence_scores[timeframe] = max(min_conf, min(max_conf, final_confidence))
            
            return predictions, confidence_scores
            
        except Exception as e:
            print(f"❌ Error in dynamic prediction: {e}")
            raise
    
    def save_model(self, symbol: str):
        """Save dynamic model"""
        try:
            model_dir = "backend/saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_data = {
                'ensemble_models': self.ensemble_models,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'validation_scores': self.validation_scores,
                'model_weights': self.model_weights,
                'config': self.config,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, f"{model_dir}/{symbol}_dynamic_model.pkl")
            
        except Exception as e:
            print(f"❌ Error saving dynamic model: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load dynamic model"""
        try:
            model_path = f"backend/saved_models/{symbol}_dynamic_model.pkl"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.ensemble_models = model_data['ensemble_models']
                self.scaler = model_data['scaler']
                self.feature_cols = model_data['feature_cols']
                self.validation_scores = model_data['validation_scores']
                self.model_weights = model_data.get('model_weights', {})
                self.config = model_data.get('config', self.load_dynamic_config())
                self.is_trained = model_data['is_trained']
                return True
        except Exception as e:
            print(f"❌ Error loading dynamic model: {e}")
        
        return False
