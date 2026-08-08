import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller


# Load the data
data = pd.read_csv("Time_seriestechnic/TimeSeries_TotalSolarGen_and_Load_IT_2016.csv")
print(data.head())

# Visualize the data 

# Convert utc_timestamp to datetime
data['utc_timestamp'] = pd.to_datetime(data['utc_timestamp'])

# Plot the data
plt.figure(figsize=(14,6))

plt.plot(data['utc_timestamp'], data['IT_load_new'], label='Load')
plt.plot(data['utc_timestamp'], data['IT_solar_generation'], label='Solar Generation')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Load and Solar Generation over Time')
plt.show()

# Check for missing values
print(data.isnull().sum())

# Fill in missing value using forward fill
data['IT_load_new']=data['IT_load_new'].ffill()

# Check for the missing values again
print(data.isnull().sum())

# Function to perform Augmented Dickey-Fuller test
def adf_test(timerseries):
    print("Results of Dickey-Fuller test")
    dftest = adfuller(timerseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observation Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)

# Perform test for 'IT_load_new'
print("ADF test for 'IT_load_new': ")
adf_test(data['IT_load_new'])

# perform test for IT_solar_generation
print("\nADF test for 'IT_solar_generation': ")
adf_test(data['IT_solar_generation'])

## For 'IT_load_new': The p-value is extremely small (much less than 0.05), so we reject the null hypothesis that the time series id non_stationary
## Therefore, 'IT_load_new' can be considered stationary

## ## For 'IT_solar_generation': The p-value is extremely small (much less than 0.05), so we reject the null hypothesis that the time series id non_stationary

# ARIMA, which stands for AutoRegressive integrated Mving Average, is a class of models that explains a given time series based on its own
# past values, that is, own lags and the lagged forecast errors. the equation  can be used to forecast future values.
# Any 'non-seasonal' time series that exhibits patterns and is not a random white noisecan be modeled with ARIMA models
# An ARIMA model is characterized by 3 terms: p,d,q 

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
plot_acf(data['IT_load_new'], lags=50, zero=False, ax=ax1)
plot_pacf(data['IT_solar_generation'], lags=50, zero=False, ax=ax2)
plt.show()

# from the ACF plot, we see gradual decrease, and from the PACF plot, there is a sharp drop after lag 2. so we can take p=2 and q=2 our model
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Split the data into training and test sets
train_size = int(len(data['IT_load_new']) * 0.8)
train, test = data['IT_load_new'][:train_size], data['IT_load_new'][train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(2, 0, 2))
model_fit = model.fit()

# Make predictions on the test set
# Use index-based prediction to avoid start/end mismatch
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

# Calculate RMSE
rmse = sqrt(mean_squared_error(test, predictions))
print(f"Mean Square Error: {rmse:.3f}")

# Plot actual vs Predicted vcalue
plt.figure(figsize=(14,6))
plt.plot(data['utc_timestamp'][train_size:], test, label='Actual')
plt.plot(data['utc_timestamp'][train_size:],predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Actual vs Predicted load values')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
plot_acf(data['IT_solar_generation'], lags=50, zero=False, ax=ax1)
plot_pacf(data['IT_solar_generation'], lags=50, zero=False, ax=ax2)
plt.show()