import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 1. Download Data
stock = 'AAPL'
df = yf.download(stock, start='2015-01-01', end='2024-01-01')
data = df[['Close']].values

# 2. Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 3. Create sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i+time_step])
        y.append(dataset[i+time_step])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 4. Split Data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train Model
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# 7. Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
real = scaler.inverse_transform(y_test.reshape(-1, 1))

# 8. Plot
plt.figure(figsize=(12,6))
plt.plot(real, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
