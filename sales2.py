import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv('sales_data.csv', parse_dates=['date'], index_col='date')

# Data Cleaning
data.dropna(inplace=True)

# Exploratory Data Analysis
plt.figure(figsize=(10, 5))
plt.plot(data['sales'])
plt.title('Sales Over Time')
plt.show()

# Fit ARIMA Model
model = ARIMA(data['sales'], order=(1, 1, 1))
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=12) # Forecasting next 12 periods
plt.plot(data['sales'], label='Historical Sales')
plt.plot(forecast, label='Forecasted Sales', color='red')
plt.title('Sales Forecasting')
plt.legend()
plt.show()

# Calculate and print error metrics
mae = mean_absolute_error(data['sales'][-12:], forecast)
print(f'Mean Absolute Error: {mae}')
