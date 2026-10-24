import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
data = pd.read_csv("rnn_sales_data.csv")
data.head()

sales = data["Sales"].values
x =[]
y = []

for i in range(len(sales) - 3):
  x.append(sales[i:i+3])
  y.append(sales[i+3])
x = np.array(x)
y = np.array(y)
x = x.reshape((x.shape[0], x.shape[1],1))
model = Sequential([
    SimpleRNN(10, activation="relu", input_shape=(3,1)),
  Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(x,y, epochs=200, verbose=0)
prediction = model.predict(x[-1].reshape(1,3,1), verbose=0)
print("Previous 3 sales:", x[-1].flatten())
print("Actual next sales:", y[-1])
print("Predicted next sales:", round(prediction[0][0], 2))