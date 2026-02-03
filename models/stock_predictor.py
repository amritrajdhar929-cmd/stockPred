import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        
        # Technical indicators
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['volatility'] = df['close'].rolling(window=10).std()
        
        # Price changes
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # High-Low spread
        df['spread'] = (df['high'] - df['low']) / df['close']
        
        # Create feature matrix (drop NaN values)
        feature_columns = [
            'ma_5', 'ma_10', 'ma_20', 'rsi', 'volatility',
            'price_change_1', 'price_change_5', 'price_change_10',
            'volume_ratio', 'spread'
        ]
        
        feature_df = df[feature_columns].dropna()
        
        return feature_df.values
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train(self, symbol: str, df: pd.DataFrame):
        """Train the model on historical data"""
        try:
            # Prepare features
            X = self.prepare_features(df)
            
            # Prepare target variables (future prices)
            y_1_day = df['close'].shift(-1).dropna()
            y_5_day = df['close'].shift(-5).dropna()
            y_30_day = df['close'].shift(-30).dropna()
            
            # Align features and targets
            min_length = min(len(X), len(y_1_day), len(y_5_day), len(y_30_day))
            X = X[:min_length]
            y_1_day = y_1_day.iloc[:min_length]
            y_5_day = y_5_day.iloc[:min_length]
            y_30_day = y_30_day.iloc[:min_length]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models for different timeframes
            self.model_1d = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_5d = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_30d = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.model_1d.fit(X_scaled, y_1_day)
            self.model_5d.fit(X_scaled, y_5_day)
            self.model_30d.fit(X_scaled, y_30_day)
            
            self.is_trained = True
            
            # Save model
            self.save_model(symbol)
            
        except Exception as e:
            print(f"Error training model for {symbol}: {e}")
            self.is_trained = False
    
    def predict(self, df: pd.DataFrame) -> dict:
        """Make predictions for different timeframes"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            # Get latest features
            X = self.prepare_features(df)
            X_scaled = self.scaler.transform(X[-1:])  # Use latest data point
            
            # Make predictions
            pred_1d = self.model_1d.predict(X_scaled)[0]
            pred_5d = self.model_5d.predict(X_scaled)[0]
            pred_30d = self.model_30d.predict(X_scaled)[0]
            
            # Calculate confidence scores based on model uncertainty
            confidence_1d = self.calculate_confidence(self.model_1d, X_scaled)
            confidence_5d = self.calculate_confidence(self.model_5d, X_scaled)
            confidence_30d = self.calculate_confidence(self.model_30d, X_scaled)
            
            return {
                '1_day': float(pred_1d),
                '5_day': float(pred_5d),
                '30_day': float(pred_30d)
            }, {
                '1_day': confidence_1d,
                '5_day': confidence_5d,
                '30_day': confidence_30d
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def calculate_confidence(self, model, X) -> float:
        """Calculate confidence score based on tree variance"""
        try:
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X)[0] for tree in model.estimators_])
            
            # Calculate standard deviation as uncertainty measure
            std_dev = np.std(tree_predictions)
            
            # Convert to confidence score (higher confidence for lower std dev)
            confidence = max(50, min(95, 90 - std_dev * 10))
            
            return confidence
            
        except Exception:
            return 75  # Default confidence
    
    def save_model(self, symbol: str):
        """Save trained model"""
        try:
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump({
                'model_1d': self.model_1d,
                'model_5d': self.model_5d,
                'model_30d': self.model_30d,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }, f"{model_dir}/{symbol}_model.pkl")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load trained model"""
        try:
            model_path = f"saved_models/{symbol}_model.pkl"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.model_1d = model_data['model_1d']
                self.model_5d = model_data['model_5d']
                self.model_30d = model_data['model_30d']
                self.scaler = model_data['scaler']
                self.is_trained = model_data['is_trained']
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return False
