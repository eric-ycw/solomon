from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import sys
sys.path.insert(0, '..')
import os
from src.regression.linear_regression import LinearRegression
from src.utils import *

# Using multivariate linear regression to predict the close price of SPY

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/spy_regression.csv"))

# We convert the date entries to datetime and fix them in place as an index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)
df = df.sort_values('Date',ascending=True)

# We generate some test features, including:
# 1. 5-day simple moving average
# 2. 5-day standard deviation
# 3. Daily high-low % change
# 4. Daily volume % change

df['SMA_5'] = df['Adj Close'].rolling(5).mean()
df['STD_5'] = df['Adj Close'].rolling(5).std()
df['High-low pct_change'] = (df['High'] - df['Low']).pct_change()
df['Volume pct_change'] = df['Volume'].pct_change()

df = df.dropna()

# Select the feature/target value columns, do a train-test split
X = df.iloc[:,6:]
y = df.iloc[:,5:6]
X_train, y_train = X.sample(frac=0.8, random_state=20), y.sample(frac=0.8, random_state=20)
X_test, y_test = X.drop(X_train.index), y.drop(y_train.index)

# Convert into ndarrays
X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

mean_norm = MeanNormalization()
X_train, X_test = mean_norm.normalize(X_train), mean_norm.normalize(X_test)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# We initialize weights as zeros
weights = np.zeros((X_train.shape[1], 1))

lr = LinearRegression()
weights, loss_hist, loss = lr.train(X_train, y_train, weights, display=True)

y_hat, loss = lr.predict(X_test, y_test, weights)
print('Testing set loss: ', loss)

plt.title('Loss against epoch')
plt.plot(np.arange(1, loss_hist.shape[0] + 1), loss_hist)
plt.show()
