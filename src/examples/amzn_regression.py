from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as skl_LR
from sklearn.metrics import r2_score
plt.style.use('seaborn')

import sys
sys.path.insert(0, '..')
import os
from src.regression.linear_regression import LinearRegression
from src.utils import *

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/AMZN.csv"))
spy = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/SPY.csv"))
wmt = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/WMT.csv"))
ebay = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/EBAY.csv"))
aapl = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/AAPL.csv"))

df.drop(df.columns[[1, 2, 3, 4, 6]], axis=1, inplace=True)
df.rename(columns={'Adj Close' : 'AMZN'}, inplace=True)
df['SPY'] = spy['Adj Close']
df['WMT'] = wmt['Adj Close']
df['EBAY'] = ebay['Adj Close']
df['AAPL'] = aapl['Adj Close']

# We convert the date entries to datetime and fix them in place as an index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print(df)
plt.title('Correlation matrix')
sns.heatmap(df.corr(), annot=True)
plt.show()

# Select the feature/target value columns, do a train-test split
X = df.iloc[:,1:]
y = df.iloc[:,0:1]
X_train, y_train = X.sample(frac=0.8, random_state=20), y.sample(frac=0.8, random_state=20)
X_test, y_test = X.drop(X_train.index), y.drop(y_train.index)

X_test_spy = X_test['SPY']

# Convert into ndarrays
X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

# We use mean normalization to rescale the features set
mean_norm = MeanNormalization()
X_train, X_test = mean_norm.normalize(X_train), mean_norm.normalize(X_test)

# Add a column of ones to the features set
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Train our weights and get the loss history
lr = LinearRegression()
loss_hist, loss = lr.train(X_train, y_train, display=True)

# Predict the close prices for the test set and calculate our loss and R-squared
y_hat, loss = lr.predict(X_test, y_test)
print('Testing set loss:', loss)
r_squared = r2_score(y_test, y_hat)
print('Our R-squared:', r_squared)

# Train and fit using sklearn
skl_lr = skl_LR()
skl_lr.fit(X_train, y_train)
skl_score = skl_lr.score(X_test, y_test)
print('sklearn\'s R-squared:', skl_score)

print('Weights:', np.squeeze(lr.params.T))

# Plot loss history
plt.title('Loss against iterations')
plt.plot(np.arange(1, loss_hist.shape[0] + 1), loss_hist)
plt.xlabel('Iterations')
plt.ylabel('Loss (MSE)')
plt.show()

# Plot predicted vs actual SPY closing prices
y_test, y_hat = np.squeeze(y_test.T), np.squeeze(y_hat.T)
plt.title('Predicted vs Actual AMZN Price')
plt.plot(np.arange(1, y_test.shape[0] + 1), y_test, label="Actual")
plt.plot(np.arange(1, y_hat.shape[0] + 1), y_hat, label="Predicted")
plt.plot(np.arange(1, X_test_spy.shape[0] + 1), X_test_spy, label="SPY")
plt.legend(loc="upper left")
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()
