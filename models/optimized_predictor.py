import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptimizedPredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.ensemble_models = {}
        self.feature_importance = {}
        self.is_trained = False
        self.validation_scores = {}
        self.model_weights = {}
        self.config = self.load_optimized_config()
        
    def load_optimized_config(self):
        """Load optimized configuration for better predictions"""
        return {
            'prediction_timeframes': [1, 5, 30],
            'min_confidence_thresholds': {'1_day': 85, '5_day': 83, '30_day': 80},
            'max_confidence_thresholds': {'1_day': 96, '5_day': 94, '30_day': 92},
            'ensemble_models': ['xgb', 'rf', 'gb', 'et'],
            'feature_periods': {
                'returns': [1, 2, 3, 5, 7, 10, 14, 21, 30],
                'ma': [5, 10, 20, 50, 100],
                'volatility': [5, 10, 20, 30],
                'volume': [5, 10, 20],
                'lags': [1, 2, 3, 5, 7, 10, 14]
            },
            'optimized_params': {
                'xgb': {
                    'n_estimators': 500,
                    'max_depth': 10,
                    'learning_rate': 0.02,
                    'subsample': 0.95,
                    'colsample_bytree': 0.95,
                    'reg_alpha': 0.05,
                    'reg_lambda': 1.5,
                    'min_child_weight': 1,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'rf': {
                    'n_estimators': 300,
                    'max_depth': 20,
                    'min_samples_split': 3,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'gb': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.03,
                    'subsample': 0.85,
                    'max_features': 'sqrt',
                    'min_samples_split': 3,
                    'random_state': 42
                },
                'et': {
                    'n_estimators': 250,
                    'max_depth': 15,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': False,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'timeframe_specific_params': {
                '1_day': {
                    'lookback_period': 60,
                    'feature_importance_threshold': 0.01
                },
                '5_day': {
                    'lookback_period': 90,
                    'feature_importance_threshold': 0.008
                },
                '30_day': {
                    'lookback_period': 180,
                    'feature_importance_threshold': 0.005
                }
            }
        }
    
    def prepare_optimized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare optimized features for better prediction accuracy"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # ===== ENHANCED RETURNS =====
        for period in self.config['feature_periods']['returns']:
            df[f'returns_{period}d'] = df['close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
            df[f'returns_{period}d_abs'] = abs(df[f'returns_{period}d'])
        
        # Cumulative returns with different windows
        df['cum_return_3d'] = df['returns_1d'].rolling(3).sum()
        df['cum_return_5d'] = df['returns_1d'].rolling(5).sum()
        df['cum_return_10d'] = df['returns_1d'].rolling(10).sum()
        df['cum_return_20d'] = df['returns_1d'].rolling(20).sum()
        
        # Rolling max/min returns
        df['max_return_5d'] = df['returns_1d'].rolling(5).max()
        df['min_return_5d'] = df['returns_1d'].rolling(5).min()
        df['max_return_20d'] = df['returns_1d'].rolling(20).max()
        df['min_return_20d'] = df['returns_1d'].rolling(20).min()
        
        # ===== ENHANCED MOVING AVERAGES =====
        for period in self.config['feature_periods']['ma']:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
            df[f'price_above_sma_{period}'] = (df['close'] > df[f'sma_{period}']).astype(int)
            df[f'price_above_ema_{period}'] = (df['close'] > df[f'ema_{period}']).astype(int)
        
        # Enhanced EMA system (after basic EMAs are created)
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # EMA ratios and crossovers
        df['ema_8_12_ratio'] = df['ema_8'] / df['ema_12']
        df['ema_12_26_ratio'] = df['ema_12'] / df['ema_26']
        df['ema_26_50_ratio'] = df['ema_26'] / df['ema_50']
        
        df['ema_8_12_cross'] = (df['ema_8'] > df['ema_12']).astype(int)
        df['ema_12_26_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
        df['ema_26_50_cross'] = (df['ema_26'] > df['ema_50']).astype(int)
        
        # SMA crossovers
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['sma_50_100_cross'] = (df['sma_50'] > df['sma_100']).astype(int)
        
        # ===== ENHANCED TECHNICAL INDICATORS =====
        # RSI with multiple periods
        for period in [7, 14, 21, 30]:
            df[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
            df[f'rsi_{period}_normalized'] = df[f'rsi_{period}'] / 100
        
        # RSI levels
        df['rsi_14_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_14_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_14_neutral'] = ((df['rsi_14'] >= 30) & (df['rsi_14'] <= 70)).astype(int)
        
        # Enhanced MACD system
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_bullish'] = (df['macd_histogram'] > 0).astype(int)
        
        # MACD with different parameters
        df['macd_fast'] = df['ema_8'] - df['ema_21']
        df['macd_fast_signal'] = df['macd_fast'].ewm(span=7).mean()
        df['macd_fast_histogram'] = df['macd_fast'] - df['macd_fast_signal']
        
        # Enhanced Bollinger Bands
        for std in [1.5, 2.0, 2.5]:
            df[f'bb_upper_{std}'] = df['sma_20'] + std * df['close'].rolling(20).std()
            df[f'bb_lower_{std}'] = df['sma_20'] - std * df['close'].rolling(20).std()
            df[f'bb_width_{std}'] = (df[f'bb_upper_{std}'] - df[f'bb_lower_{std}']) / df['sma_20']
            df[f'bb_position_{std}'] = (df['close'] - df[f'bb_lower_{std}']) / (df[f'bb_upper_{std}'] - df[f'bb_lower_{std}'])
            df[f'bb_upper_touch_{std}'] = (df['close'] >= df[f'bb_upper_{std}']).astype(int)
            df[f'bb_lower_touch_{std}'] = (df['close'] <= df[f'bb_lower_{std}']).astype(int)
        
        # Bollinger Band squeeze
        df['bb_squeeze'] = (df['bb_width_2.0'] < df['bb_width_2.0'].rolling(50).mean()).astype(int)
        
        # ===== ENHANCED VOLATILITY FEATURES =====
        for period in self.config['feature_periods']['volatility']:
            df[f'volatility_{period}d'] = df['close'].rolling(period).std()
            df[f'volatility_pct_{period}d'] = df['close'].rolling(period).std() / df['close'].rolling(period).mean()
            df[f'volatility_ratio_{period}d'] = df[f'volatility_{period}d'] / df[f'volatility_{period}d'].rolling(50).mean()
        
        # Enhanced ATR
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = self.calculate_atr(df, period)
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close'].rolling(period).mean()
        
        # Volatility regime (using percentiles)
        vol_20d = df['volatility_20d']
        df['volatility_regime'] = np.where(vol_20d <= vol_20d.quantile(0.25), 0,
                                         np.where(vol_20d <= vol_20d.quantile(0.75), 1, 2))
        
        # Volatility breakout
        df['volatility_breakout'] = (df['volatility_5d'] > df['volatility_20d'] * 1.5).astype(int)
        
        # ===== ENHANCED VOLUME FEATURES =====
        for period in self.config['feature_periods']['volume']:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ema_{period}'] = df['volume'].ewm(span=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
            df[f'volume_ratio_ema_{period}'] = df['volume'] / df[f'volume_ema_{period}']
        
        # Volume patterns
        df['volume_spike'] = (df['volume'] > df['volume_sma_20'] * 2).astype(int)
        df['volume_dry_up'] = (df['volume'] < df['volume_sma_20'] * 0.5).astype(int)
        
        # Enhanced Volume Price Trend
        df['vpt'] = self.calculate_vpt(df)
        df['vpt_sma_10'] = df['vpt'].rolling(10).mean()
        df['vpt_sma_20'] = df['vpt'].rolling(20).mean()
        df['vpt_ratio_10'] = df['vpt'] / df['vpt_sma_10']
        df['vpt_ratio_20'] = df['vpt'] / df['vpt_sma_20']
        
        # On-Balance Volume
        df['obv'] = self.calculate_obv(df)
        df['obv_sma_10'] = df['obv'].rolling(10).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma_10']
        
        # Enhanced VWAP
        for period in [10, 20, 50]:
            df[f'vwap_{period}'] = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
            df[f'price_vwap_ratio_{period}'] = df['close'] / df[f'vwap_{period}']
        
        # ===== ENHANCED PRICE PATTERN FEATURES =====
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['high'] - df['low'])
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['high'] - df['low'])
        df['real_body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Enhanced candlestick patterns
        df['doji'] = (abs(df['close'] - df['open']) / df['open'] < 0.005).astype(int)
        df['hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & (df['upper_shadow'] < 0.1 * df['body_size'])).astype(int)
        df['shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & (df['lower_shadow'] < 0.1 * df['body_size'])).astype(int)
        df['engulfing_bullish'] = ((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & 
                                  (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))).astype(int)
        
        # Gap analysis
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_up_size'] = np.where(df['gap_up'] == 1, df['gap_size'], 0)
        df['gap_down_size'] = np.where(df['gap_down'] == 1, abs(df['gap_size']), 0)
        
        # ===== ENHANCED TIME FEATURES =====
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
        
        # Seasonal indicators
        df['is_winter'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_monsoon'] = ((df['month'] >= 7) & (df['month'] <= 9)).astype(int)
        
        # Trading session indicators
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        
        # ===== ENHANCED LAG FEATURES =====
        for lag in self.config['feature_periods']['lags']:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns_1d'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
            df[f'atr_14_lag_{lag}'] = df['atr_14'].shift(lag)
        
        # Rolling statistics with different windows
        for period in [3, 5, 10, 20]:
            df[f'returns_mean_{period}d'] = df['returns_1d'].rolling(period).mean()
            df[f'returns_std_{period}d'] = df['returns_1d'].rolling(period).std()
            df[f'returns_skew_{period}d'] = df['returns_1d'].rolling(period).skew()
            df[f'returns_kurt_{period}d'] = df['returns_1d'].rolling(period).kurt()
        
        # ===== ENHANCED MARKET MICROSTRUCTURE =====
        df['price_efficiency'] = abs(df['close'] - df['vwap_20']) / df['vwap_20']
        df['liquidity_ratio'] = df['volume'] / (df['high'] - df['low'])
        df['depth_ratio'] = (df['high'] - df['low']) / df['close']
        df['spread_ratio'] = (df['high'] - df['low']) / df['vwap_20']
        
        # Price momentum indicators
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Rate of change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with enhanced handling"""
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
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = np.where(df['close'] > df['close'].shift(), df['volume'],
                      np.where(df['close'] < df['close'].shift(), -df['volume'], 0)).cumsum()
        return pd.Series(obv, index=df.index)
    
    def prepare_features_for_training(self, df: pd.DataFrame, timeframe: str) -> tuple:
        """Prepare features and targets for specific timeframe"""
        # Get timeframe-specific configuration
        tf_config = self.config['timeframe_specific_params'][timeframe]
        lookback_period = tf_config['lookback_period']
        
        # Get all feature columns (exclude target and date)
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values
        feature_cols = [col for col in feature_cols if df[col].isna().sum() < len(df) * 0.3]
        
        # Prepare features with specific lookback period
        X = df[feature_cols].copy()
        
        # Handle NaN values
        X = X.fillna(method='ffill').fillna(X.mean())
        
        # Prepare target for specific timeframe
        days = int(timeframe.split('_')[0])
        y = df['close'].shift(-days)
        
        # Remove rows with NaN targets
        valid_mask = y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        # Ensure minimum data points
        if len(X_valid) < lookback_period:
            print(f"⚠️ Insufficient data for {timeframe}: {len(X_valid)} samples, need {lookback_period}")
            return None, None, None
        
        return X_valid, y_valid, feature_cols
    
    def create_optimized_ensemble(self):
        """Create optimized ensemble with all models"""
        estimators = []
        
        if 'xgb' in self.config['ensemble_models']:
            xgb_model = xgb.XGBRegressor(**self.config['optimized_params']['xgb'])
            estimators.append(('xgb', xgb_model))
        
        if 'rf' in self.config['ensemble_models']:
            rf_model = RandomForestRegressor(**self.config['optimized_params']['rf'])
            estimators.append(('rf', rf_model))
        
        if 'gb' in self.config['ensemble_models']:
            gb_model = GradientBoostingRegressor(**self.config['optimized_params']['gb'])
            estimators.append(('gb', gb_model))
        
        if 'et' in self.config['ensemble_models']:
            et_model = ExtraTreesRegressor(**self.config['optimized_params']['et'])
            estimators.append(('et', et_model))
        
        return VotingRegressor(estimators)
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='xgb'):
        """Optimize hyperparameters for better performance"""
        if model_type == 'xgb':
            param_grid = {
                'n_estimators': [300, 500, 700],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.02, 0.03],
                'subsample': [0.8, 0.9, 0.95]
            }
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        elif model_type == 'rf':
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 3, 5]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        else:
            return None
        
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_
    
    def train_optimized_model(self, symbol: str, df: pd.DataFrame):
        """Train optimized model for better predictions"""
        try:
            print(f"🚀 Training optimized model for {symbol}...")
            
            # Prepare optimized features
            df_features = self.prepare_optimized_features(df)
            
            # Train models for each timeframe
            for timeframe in ['1_day', '5_day', '30_day']:
                print(f"   Training {timeframe} optimized model for {symbol}...")
                
                # Prepare features for specific timeframe
                X, y, feature_cols = self.prepare_features_for_training(df_features, timeframe)
                
                if X is None or y is None:
                    print(f"❌ Insufficient data for {timeframe}")
                    continue
                
                # Scale features
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Create optimized ensemble
                ensemble_model = self.create_optimized_ensemble()
                
                # Time series split for validation
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Cross-validation
                cv_mae = cross_val_score(ensemble_model, X_scaled, y, cv=tscv, 
                                       scoring='neg_mean_absolute_error')
                cv_r2 = cross_val_score(ensemble_model, X_scaled, y, cv=tscv, 
                                      scoring='r2')
                
                # Train on full dataset
                ensemble_model.fit(X_scaled, y)
                
                # Store model, scaler, and validation scores
                self.ensemble_models[timeframe] = {
                    'model': ensemble_model,
                    'scaler': scaler,
                    'feature_cols': feature_cols
                }
                
                self.validation_scores[timeframe] = {
                    'mae': -cv_mae.mean(),
                    'mae_std': cv_mae.std(),
                    'r2': cv_r2.mean(),
                    'r2_std': cv_r2.std(),
                    'samples': len(X),
                    'features': len(feature_cols)
                }
                
                # Calculate optimized weights
                mae_weight = 1 / (-cv_mae.mean() + 1e-6)
                r2_weight = cv_r2.mean() if cv_r2.mean() > 0 else 0.1
                self.model_weights[timeframe] = (mae_weight + r2_weight) / 2
                
                print(f"   ✅ {timeframe}: MAE = {-cv_mae.mean():.4f}, R² = {cv_r2.mean():.4f}")
            
            self.is_trained = True
            
            # Save optimized model
            self.save_optimized_model(symbol)
            
            print(f"✅ Optimized model trained for {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error training optimized model for {symbol}: {e}")
            self.is_trained = False
            return False
    
    def predict_optimized(self, df: pd.DataFrame) -> tuple:
        """Make optimized predictions with high confidence"""
        if not self.is_trained:
            raise ValueError("Optimized model not trained yet")
        
        try:
            # Prepare optimized features
            df_features = self.prepare_optimized_features(df)
            
            predictions = {}
            confidence_scores = {}
            
            for timeframe in ['1_day', '5_day', '30_day']:
                if timeframe in self.ensemble_models:
                    model_data = self.ensemble_models[timeframe]
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_cols = model_data['feature_cols']
                    
                    # Prepare features for prediction
                    X = df_features[feature_cols].copy()
                    X = X.fillna(method='ffill').fillna(X.mean())
                    
                    # Get the latest data point
                    X_latest = X.iloc[-1:].values
                    X_scaled = scaler.transform(X_latest)
                    
                    # Get prediction
                    pred = model.predict(X_scaled)[0]
                    predictions[timeframe] = float(pred)
                    
                    # Calculate optimized confidence
                    val_score = self.validation_scores[timeframe]
                    current_price = df['close'].iloc[-1]
                    
                    # Enhanced confidence calculation
                    mae_ratio = val_score['mae'] / current_price
                    base_confidence = max(80, min(98, 100 - (mae_ratio * 1000)))
                    
                    # Boost confidence based on R² score
                    r2_boost = min(15, val_score['r2'] * 30)
                    
                    # Additional confidence from ensemble stability
                    ensemble_stability = min(10, self.model_weights[timeframe] * 20)
                    
                    # Sample size confidence
                    sample_confidence = min(5, (val_score['samples'] / 1000) * 5)
                    
                    # Final confidence calculation
                    final_confidence = min(96, base_confidence + r2_boost + ensemble_stability + sample_confidence)
                    
                    # Apply optimized confidence thresholds
                    min_conf = self.config['min_confidence_thresholds'][timeframe]
                    max_conf = self.config['max_confidence_thresholds'][timeframe]
                    confidence_scores[timeframe] = max(min_conf, min(max_conf, final_confidence))
            
            return predictions, confidence_scores
            
        except Exception as e:
            print(f"❌ Error in optimized prediction: {e}")
            raise
    
    def save_optimized_model(self, symbol: str):
        """Save optimized model"""
        try:
            model_dir = "backend/saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_data = {
                'ensemble_models': self.ensemble_models,
                'validation_scores': self.validation_scores,
                'model_weights': self.model_weights,
                'config': self.config,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, f"{model_dir}/{symbol}_optimized_model.pkl")
            
        except Exception as e:
            print(f"❌ Error saving optimized model: {e}")
    
    def load_optimized_model(self, symbol: str) -> bool:
        """Load optimized model"""
        try:
            model_path = f"backend/saved_models/{symbol}_optimized_model.pkl"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.ensemble_models = model_data['ensemble_models']
                self.validation_scores = model_data['validation_scores']
                self.model_weights = model_data.get('model_weights', {})
                self.config = model_data.get('config', self.load_optimized_config())
                self.is_trained = model_data['is_trained']
                return True
        except Exception as e:
            print(f"❌ Error loading optimized model: {e}")
        
        return False
