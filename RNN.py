# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:08:57 2020

@author: Meekey
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
#import training set
df = pd.read_csv('')
df = df.dropna()
train_set = df['Adj Close']
train_set = df.iloc[:,1:2].values
#%%
#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

train_set_scaled = sc.fit_transform(train_set)
#%%
#Creating a data structure with 60 timestamps and 1 output
x_train = []
y_train =[]

for i in range(60, len(train_set_scaled)):
    x_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#%%
#Building the RNN
#importing keras and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initializing RNN
regressor = Sequential()


#Adding first LSTM layer and some dropout regularization
#%%
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding second LSTM layer and some dropout regularization
#%%
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding third LSTM layer and some dropout regularization
#%%
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding fourth LSTM layer and some dropout regularization
#%%
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

#Adding output layer
#%%
regressor.add(Dense(units = 1))

#Compile the rnn
#%%
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN on training set
#%%
regressor.fit(x_train, y_train, epochs=50, batch_size=32)


#Getting the real stock price
#%%
df_test = pd.read_csv(')
#df_test = df_test.dropna()

real_stock_price = df_test.iloc[:, 1:2].values


#Making predictions
#%%
df_total = pd.concat((df['Open'], df_test['Open']), axis = 0)
inputs = df_total[len(df_total) - len(df_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60, len(inputs)):
    x_test.append(inputs[i-60:i,0])
    
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
pred_price_rnn = regressor.predict(x_test)
pred_price_rnn = sc.inverse_transform(pred_price_rnn)

#Visual Results
#%%
plt.plot(real_stock_price, color = 'red', label = 'real stock price')
plt.plot(pred_price_rnn, color = 'blue', label = 'pred stock price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

import yfinance as yf
























