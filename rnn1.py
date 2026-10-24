import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
data = pd.read_csv("reliance_stock.csv")
data.head()
data.tail()
price = data["Close"].values
scaler = MinMaxScaler(feature_range=(0,1))
price = scaler.fit_transform(price.reshape(-1,1))
squence_length = 5
x = []
y = []
for i in range(squence_length, len(price)):
    x.append(price[i-squence_length:i])
    y.append(price[i])
x = np.array(x)
y = np.array(y)
x[0]
y[0:5]
model = Sequential()
model.add(
    SimpleRNN(
    50,
    activation="tanh",
    input_shape=(5,1)
    )
)

model.compile(
    loss="mse",
    optimizer="adam"
)

import numpy as np
x = np.array(x)
y = np.array(y).reshape(-1,1)
model.fit(
    x,
    y,
    epochs=50,
    batch_size=8,
    verbose=1
)
predicted = model.predict(x)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y)

for i in range(10):
  print(
      "Date:", data["Date"].iloc[i+squence_length],
      "Actual Price:", round(actual[i][0], 2),
      "Predicted:", round(predicted[i][0], 2)
  )