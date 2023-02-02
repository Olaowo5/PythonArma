#ARIMA stands for Autoregressive Integrated Moving Average.
#  It is an algorithm used for forecasting Time Series Data.
#  ARIMA models have three parameters like ARIMA(p, d, q).
#  Here p, d, and q are defined as:
#p is the number of lagged values that need to be 
# added or subtracted from the values (label column). 
# It captures the autoregressive part of ARIMA.
#d represents the number of times the data needs to 
# differentiate to produce a stationary signal. 
# If it’s stationary data, the value of d should be 0, 
# and if it’s seasonal data, the value of d should be 1. 
# d captures the integrated part of ARIMA.
#q is the number of lagged values for the error term added 
# or subtracted from the values (label column). 
# It captures the moving average part of ARIMA.


#will collect Google Stock Price data using Yahoo finance
import pandas as pd
import yfinance as yf
import datetime as dt

today = dt.date.today()

d1 = today.strftime("%Y-%m-%d")
end_date =d1
d2 = dt.date.today()- dt.timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")

start_date=d2

data = yf.download('GOOG',
                    start=start_date,
                    end=end_date,
                    progress=False )   

data["Date"] = data.index
#data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
#data.reset_index(drop=True, inplace=True)
#print(data.tail())

#Get only the date and close prices columns for the rest of the 
# task,
# so selecting both the columns and move further:
data = data[["Date","Close"]]
print(data.head())

#visualize the close prices of Google before moving forward

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,10))
plt.plot(data["Date"],data["Close"])

origplt = plt
origplt.show()

#need to check if the data is stational or seasonal

#To check whether our dataset is stationary or seasonal properly,
#  we can use the seasonal decomposition method that splits the 
# time series data into trend, seasonal, 
# and residuals for a better understanding of the 
# time series data
#NOTE 
#Seems to be an issue with the following lie will 
#freeze on run but works in debug mode, Visual studio code

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data["Close"], 
                            model='multiplicative', period = 30)
#fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(15, 10)

#fig.show()

#from figure 2 the data is seasonal
#We need to use the Seasonal ARIMA (SARIMA) model for Time Series Forecasting on this data


#To use ARIMA or SARIMA, we need to find the p, d, and q values. 
# We can find the value of p by plotting the autocorrelation of the Close column 
# and the value of q by plotting the partial autocorrelation plot. 
# The value of d is either 0 or 1. If the data is stationary, 
# we should use 0,and if the data is seasonal, we should use 1. 
# As our data is seasonal, we should use 1 as the d value.

figii = pd.plotting.autocorrelation_plot(data["Close"])

print("\n Done Finding P")
#from fig value pf p is 5

#Now find value of q
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data["Close"], lags = 100)

print("\n Done Finding Q")

#from the autocorrelation plot  the value of q is 2

#designing the model

 
p, d, q = 5, 1, 2
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data["Close"], order=(p,d,q))  
fitted = model.fit()  
print("\n")
print(fitted.summary())
print("\n")

#Predict the values using the ARIMA model:
print("Prediction Model")
predictions = fitted.predict()
print(predictions)

#The predicted values are wrong because the data is seasonal.
#  ARIMA model will never perform well on seasonal time series data. 
# So, here’s how to build a SARIMA model:

import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(data['Close'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())

#Now let’s predict the future stock prices 
# using the SARIMA model for the next 10 days:
print("\n Prediction Model \n using SARIMA model")
predictions = model.predict(len(data), len(data)+10)
print(predictions)

origplt.show()
#Now Plot the predicitions using the new model
data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")

print("And Done")