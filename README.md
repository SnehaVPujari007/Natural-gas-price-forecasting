# Natural gas price forecasting 📈

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A Python-based natural gas price prediction system that analyzes historical data, identifies seasonal patterns, and forecasts future prices for trading and risk management applications.

## 🚀 Features

- **Historical Analysis**: Price trends, seasonal patterns, and volatility assessment
- **Predictive Modeling**: Polynomial regression with seasonal components for accurate forecasting
- **Interactive Visualizations**: Comprehensive charts and statistical insights
- **Future Predictions**: 12-month price extrapolation with date-specific estimates

## 📊 Quick Start

```python
from nat_gas_analyzer import NaturalGasPriceAnalyzer

# Initialize analyzer
analyzer = NaturalGasPriceAnalyzer('Nat_Gas.csv')

# Visualize data and patterns
analyzer.visualize_data()
analyzer.analyze_seasonal_patterns()

# Train model and make predictions
analyzer.train_models()
price = analyzer.predict_price('2025-06-30')
print(f"Predicted price: ${price:.2f}")

# Generate 12-month forecast
analyzer.plot_predictions(months_ahead=12)
```

## 📋 Requirements

```
pandas numpy matplotlib seaborn scikit-learn
```

## 📈 Data Format

CSV file with monthly gas prices:
```csv
Dates,Prices
10/31/20,1.01E+01
11/30/20,1.03E+01
...
```

## 🎯 Key Results

- **Model Performance**: R² = 0.847, RMSE = $0.43
- **Seasonal Insight**: Winter prices ~$1.10/MMBtu higher than summer
- **Price Range**: $9.84 - $12.80 (Oct 2020 - Sep 2024)

## 🔧 Installation

```bash
git clone https://github.com/your-username/natural-gas-price-analysis.git
cd natural-gas-price-analysis
pip install -r requirements.txt
python main.py
```

## 🏢 Business Applications

- Trading desk price forecasting
- Storage contract optimization
- Risk management and scenario planning
- Client pricing discussions

---

⭐ **Star this repo if it helps your energy trading analysis!**
