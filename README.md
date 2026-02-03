# NSE Stock Prediction AI

A comprehensive stock prediction system that uses machine learning to predict NSE stock prices for 1, 5, and 30 days with confidence scores.

## Features

- **Real-time Stock Data**: Fetches current stock prices from NSE using Yahoo Finance API
- **AI-Powered Predictions**: Uses Random Forest models trained on historical data
- **Multiple Timeframes**: Predictions for 1 day, 5 days, and 30 days
- **Confidence Scores**: Each prediction includes a confidence score (50-95%)
- **Modern Frontend**: Beautiful, responsive UI with charts and visualizations
- **30 NSE Stocks**: Covers major NSE stocks including RELIANCE, TCS, HDFCBANK, etc.
- **Synthetic Training Data**: 3 years of synthetic historical data for training

## Project Structure

```
StockPred/
├── backend/
│   ├── main.py              # FastAPI backend server
│   ├── train_models.py      # ML model training script
│   └── saved_models/        # Trained ML models
├── frontend/
│   ├── index.html           # Main frontend page
│   └── app.js              # Frontend JavaScript
├── models/
│   └── stock_predictor.py   # ML prediction model
├── data/
│   ├── generate_dataset.py  # Synthetic data generation
│   ├── *_data.csv          # Historical data for each stock
│   └── dataset_metadata.json
└── requirements.txt         # Python dependencies
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd StockPred
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Generate training data**:
```bash
python3 data/generate_dataset.py
```

4. **Train ML models**:
```bash
python3 backend/train_models.py
```

5. **Start the backend server**:
```bash
cd backend
python3 main.py
```

6. **Open the frontend**:
Open your browser and go to `http://localhost:8000/static/index.html`

## API Endpoints

### Get Available Stocks
```
GET /stocks
```
Returns a list of all available NSE stocks.

### Get Stock Prediction
```
POST /predict
Content-Type: application/json

{
  "symbol": "RELIANCE"
}
```
Returns current price, predictions, and confidence scores.

## Supported Stocks

The system supports 30 major NSE stocks:
- RELIANCE (Reliance Industries Ltd.)
- TCS (Tata Consultancy Services)
- HDFCBANK (HDFC Bank Ltd.)
- INFY (Infosys Ltd.)
- ICICIBANK (ICICI Bank Ltd.)
- And 25 more...

## Model Architecture

- **Algorithm**: Random Forest Regressor
- **Features**: Moving averages, RSI, volatility, price changes, volume ratios
- **Training Data**: 3 years of synthetic historical data
- **Timeframes**: Separate models for 1-day, 5-day, and 30-day predictions
- **Confidence Calculation**: Based on tree ensemble variance

## Frontend Features

- **Stock Search**: Autocomplete search with stock suggestions
- **Real-time Predictions**: Live price predictions with confidence scores
- **Interactive Charts**: Price prediction visualization using Chart.js
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Glassmorphism design with smooth animations

## Confidence Score Interpretation

- **80-95%**: High confidence - Very reliable prediction
- **65-79%**: Medium confidence - Moderately reliable prediction  
- **50-64%**: Low confidence - Less reliable prediction

## Technical Details

### Data Fetching
- Primary: Yahoo Finance API (free, reliable)
- Fallback: Synthetic price generation based on stock symbol hash

### Prediction Process
1. Load trained ML model for the specific stock
2. If model unavailable, use statistical fallback
3. Calculate technical indicators (MA, RSI, volatility)
4. Generate predictions for each timeframe
5. Calculate confidence scores based on model uncertainty

### Error Handling
- Graceful fallback when API fails
- Synthetic data generation for missing stocks
- Comprehensive error logging

## Usage Example

1. Open the web interface
2. Search for a stock (e.g., "RELIANCE")
3. View current price and AI predictions
4. Check confidence scores for each prediction
5. Analyze the prediction chart

## Disclaimer

⚠️ **Important**: This system is for educational purposes only. Stock predictions are not financial advice. Always consult with qualified financial professionals before making investment decisions.

## Development

### Adding New Stocks
1. Add the stock symbol to `NSE_STOCKS` list in `backend/main.py`
2. Regenerate the dataset: `python3 data/generate_dataset.py`
3. Retrain models: `python3 backend/train_models.py`

### Customizing Models
Edit `models/stock_predictor.py` to:
- Change the ML algorithm
- Add new technical indicators
- Modify confidence calculation logic

## Performance

- **Training Time**: ~2-3 minutes for all 30 stocks
- **Prediction Time**: <100ms per request
- **Memory Usage**: ~500MB for all models
- **Accuracy**: Synthetic dataset - designed for demonstration

## License

MIT License - Feel free to use and modify for educational purposes.
