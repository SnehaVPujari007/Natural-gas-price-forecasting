import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NaturalGasPriceAnalyzer:
    def __init__(self, csv_file_path):
        """
        Initialize the analyzer with natural gas price data
        """
        self.data = None
        self.model = None
        self.seasonal_model = None
        self.load_data(csv_file_path)
        self.prepare_features()
        
    def load_data(self, csv_file_path):
        """
        Load and clean the natural gas price data
        """
        # Read the CSV file
        df = pd.read_csv('Nat_Gas.csv')
        
        # Convert dates to datetime
        df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
        
        # Convert scientific notation prices to float
        df['Prices'] = df['Prices'].astype(float)
        
        # Sort by date
        df = df.sort_values('Dates').reset_index(drop=True)
        
        self.data = df
        print(f"Loaded {len(df)} data points from {df['Dates'].min().strftime('%Y-%m-%d')} to {df['Dates'].max().strftime('%Y-%m-%d')}")
        
    def prepare_features(self):
        """
        Create features for modeling including seasonal and trend components
        """
        df = self.data.copy()
        
        # Create time-based features
        df['Year'] = df['Dates'].dt.year
        df['Month'] = df['Dates'].dt.month
        df['Quarter'] = df['Dates'].dt.quarter
        df['DayOfYear'] = df['Dates'].dt.dayofyear
        
        # Create numerical time index (months since start)
        start_date = df['Dates'].min()
        df['MonthsFromStart'] = ((df['Dates'] - start_date).dt.days / 30.44).round().astype(int)
        
        # Create seasonal features (sine/cosine for cyclical patterns)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        self.data = df
        
    def visualize_data(self):
        """
        Create comprehensive visualizations of the natural gas price data
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Natural Gas Price Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time series plot
        axes[0, 0].plot(self.data['Dates'], self.data['Prices'], 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Natural Gas Prices Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Seasonal patterns (by month)
        monthly_avg = self.data.groupby('Month')['Prices'].agg(['mean', 'std']).reset_index()
        axes[0, 1].bar(monthly_avg['Month'], monthly_avg['mean'], 
                      yerr=monthly_avg['std'], capsize=5, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Average Prices by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Price ($)')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Yearly trends
        yearly_avg = self.data.groupby('Year')['Prices'].agg(['mean', 'std']).reset_index()
        axes[1, 0].bar(yearly_avg['Year'], yearly_avg['mean'], 
                      yerr=yearly_avg['std'], capsize=5, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('Average Prices by Year')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Average Price ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Price distribution
        axes[1, 1].hist(self.data['Prices'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].axvline(self.data['Prices'].mean(), color='red', linestyle='--', 
                          label=f'Mean: ${self.data["Prices"].mean():.2f}')
        axes[1, 1].axvline(self.data['Prices'].median(), color='orange', linestyle='--', 
                          label=f'Median: ${self.data["Prices"].median():.2f}')
        axes[1, 1].set_title('Price Distribution')
        axes[1, 1].set_xlabel('Price ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print key statistics
        print("\n=== KEY STATISTICS ===")
        print(f"Average Price: ${self.data['Prices'].mean():.2f}")
        print(f"Median Price: ${self.data['Prices'].median():.2f}")
        print(f"Standard Deviation: ${self.data['Prices'].std():.2f}")
        print(f"Min Price: ${self.data['Prices'].min():.2f} on {self.data.loc[self.data['Prices'].idxmin(), 'Dates'].strftime('%Y-%m-%d')}")
        print(f"Max Price: ${self.data['Prices'].max():.2f} on {self.data.loc[self.data['Prices'].idxmax(), 'Dates'].strftime('%Y-%m-%d')}")
        
    def analyze_seasonal_patterns(self):
        """
        Analyze seasonal patterns in natural gas prices
        """
        print("\n=== SEASONAL ANALYSIS ===")
        
        # Monthly analysis
        monthly_stats = self.data.groupby('Month')['Prices'].agg(['mean', 'std', 'min', 'max']).round(2)
        monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print("\nMonthly Price Statistics:")
        print(monthly_stats)
        
        # Identify peak seasons
        highest_months = monthly_stats['mean'].nlargest(3)
        lowest_months = monthly_stats['mean'].nsmallest(3)
        
        print(f"\nHighest average prices: {', '.join(highest_months.index)} (${highest_months.mean():.2f})")
        print(f"Lowest average prices: {', '.join(lowest_months.index)} (${lowest_months.mean():.2f})")
        
        # Seasonal volatility
        winter_months = self.data[self.data['Month'].isin([12, 1, 2])]
        summer_months = self.data[self.data['Month'].isin([6, 7, 8])]
        
        print(f"\nWinter (Dec-Feb) average: ${winter_months['Prices'].mean():.2f} ± ${winter_months['Prices'].std():.2f}")
        print(f"Summer (Jun-Aug) average: ${summer_months['Prices'].mean():.2f} ± ${summer_months['Prices'].std():.2f}")
        
    def train_models(self):
        """
        Train multiple models for price prediction
        """
        print("\n=== MODEL TRAINING ===")
        
        # Prepare features
        X = self.data[['MonthsFromStart', 'Month_Sin', 'Month_Cos']].values
        y = self.data['Prices'].values
        
        # Split data (use last 20% for validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train polynomial regression model
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('linear', LinearRegression())
        ])
        
        self.model.fit(X_train, y_train)
        
        # Validate model
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Model Performance (Validation Set):")
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: ${np.sqrt(mse):.2f}")
        print(f"Mean Absolute Error: ${np.mean(np.abs(y_val - y_pred)):.2f}")
        
    def predict_price(self, target_date):
        """
        Predict natural gas price for a given date
        
        Args:
            target_date (str or datetime): Date for prediction (format: 'YYYY-MM-DD' or datetime object)
            
        Returns:
            float: Predicted price
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Calculate features for target date
        start_date = self.data['Dates'].min()
        months_from_start = ((target_date - start_date).days / 30.44)
        month = target_date.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Create feature array
        features = np.array([[months_from_start, month_sin, month_cos]])
        
        # Make prediction
        predicted_price = self.model.predict(features)[0]
        
        return predicted_price
    
    def generate_future_predictions(self, months_ahead=12):
        """
        Generate predictions for the next N months
        """
        print(f"\n=== FUTURE PREDICTIONS (Next {months_ahead} months) ===")
        
        last_date = self.data['Dates'].max()
        future_dates = []
        future_predictions = []
        
        current_year = last_date.year
        current_month = last_date.month
        
        for i in range(1, months_ahead + 1):
            # Calculate next month and year
            next_month = current_month + 1
            next_year = current_year
            
            if next_month > 12:
                next_month = 1
                next_year += 1
            
            # Create the last day of the next month
            if next_month == 12:
                # December has 31 days
                future_date = pd.Timestamp(year=next_year, month=next_month, day=31)
            elif next_month in [1, 3, 5, 7, 8, 10]:
                # Months with 31 days
                future_date = pd.Timestamp(year=next_year, month=next_month, day=31)
            elif next_month in [4, 6, 9, 11]:
                # Months with 30 days
                future_date = pd.Timestamp(year=next_year, month=next_month, day=30)
            else:
                # February - check for leap year
                if next_year % 4 == 0 and (next_year % 100 != 0 or next_year % 400 == 0):
                    future_date = pd.Timestamp(year=next_year, month=next_month, day=29)
                else:
                    future_date = pd.Timestamp(year=next_year, month=next_month, day=28)
            
            # Predict price
            predicted_price = self.predict_price(future_date)
            
            future_dates.append(future_date)
            future_predictions.append(predicted_price)
            
            print(f"{future_date.strftime('%Y-%m-%d')}: ${predicted_price:.2f}")
            
            # Update for next iteration
            current_month = next_month
            current_year = next_year
        
        return future_dates, future_predictions
    
    def plot_predictions(self, months_ahead=12):
        """
        Plot historical data with future predictions
        """
        future_dates, future_predictions = self.generate_future_predictions(months_ahead)
        
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(self.data['Dates'], self.data['Prices'], 'b-', linewidth=2, 
                label='Historical Prices', alpha=0.8)
        
        # Plot predictions
        plt.plot(future_dates, future_predictions, 'r--', linewidth=2, 
                label='Predicted Prices', alpha=0.8)
        
        # Add markers
        plt.scatter(self.data['Dates'], self.data['Prices'], color='blue', s=20, alpha=0.6)
        plt.scatter(future_dates, future_predictions, color='red', s=30, alpha=0.8)
        
        plt.title('Natural Gas Price History and Predictions', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def main():
    """
    Main function to demonstrate the natural gas price analyzer
    """
    print("=== NATURAL GAS PRICE ANALYSIS SYSTEM ===\n")
    
    # Initialize analyzer (replace with your CSV file path)
    analyzer = NaturalGasPriceAnalyzer('Nat_Gas.csv')
    
    # Visualize the data
    analyzer.visualize_data()
    
    # Analyze seasonal patterns
    analyzer.analyze_seasonal_patterns()
    
    # Train prediction models
    analyzer.train_models()
    
    # Generate future predictions
    analyzer.plot_predictions(months_ahead=12)
    
    # Example predictions for specific dates
    print("\n=== SPECIFIC DATE PREDICTIONS ===")
    test_dates = ['2024-12-31', '2025-06-30', '2025-12-31']
    
    for date in test_dates:
        predicted_price = analyzer.predict_price(date)
        print(f"Predicted price for {date}: ${predicted_price:.2f}")
    
    return analyzer

# Interactive price prediction function
def predict_gas_price(date_string, analyzer):
    """
    Simple function to predict price for a given date
    
    Args:
        date_string (str): Date in 'YYYY-MM-DD' format
        analyzer: Trained NaturalGasPriceAnalyzer object
        
    Returns:
        float: Predicted price
    """
    try:
        price = analyzer.predict_price(date_string)
        return price
    except Exception as e:
        print(f"Error predicting price: {e}")
        return None

if __name__ == "__main__":
    # Run the main analysis
    gas_analyzer = main()
    
    # Example of using the predictor function
    print("\n=== INTERACTIVE PREDICTION EXAMPLE ===")
    sample_date = "2025-03-31"
    predicted_price = predict_gas_price(sample_date, gas_analyzer)
    print(f"Price prediction for {sample_date}: ${predicted_price:.2f}")