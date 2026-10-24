import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load Reliance stock dataset
data = pd.read_csv("reliance_stock.csv")

# Convert stock prices from text to numbers
data["Close"] = pd.to_numeric(
    data["Close"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

# Remove invalid rows and sort dates from oldest to newest
data["Date"] = pd.to_datetime(data["Date"], format="%d-%b-%y", errors="coerce")
data = data.dropna(subset=["Date", "Close"])
data = data.sort_values("Date").reset_index(drop=True)

print("First five rows:")
print(data.head())

print("\nLast five rows:")
print(data.tail())

# Normalize closing prices
price = data["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
price = scaler.fit_transform(price)

# Prepare sequences
sequence_length = 5
x = []
y = []

for i in range(sequence_length, len(price)):
    x.append(price[i - sequence_length:i])
    y.append(price[i])

x = np.array(x)
y = np.array(y).reshape(-1, 1)

if len(x) == 0:
    raise ValueError(
        "The CSV must contain at least 6 valid stock-price rows."
    )

print("\nFirst input sequence:")
print(x[0])

print("\nFirst five target prices:")
print(y[:5])

# Build RNN model
model = Sequential([
    SimpleRNN(
        50,
        activation="tanh",
        input_shape=(sequence_length, 1)
    ),
    Dense(1)
])

model.compile(
    loss="mse",
    optimizer="adam"
)

# Train the model
model.fit(
    x,
    y,
    epochs=50,
    batch_size=8,
    verbose=1
)

# Predict stock prices
predicted = model.predict(x, verbose=0)

predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y)

# Display predictions
print("\nStock Price Predictions:")

for i in range(min(10, len(actual))):
    print(
        "Date:", data["Date"].iloc[i + sequence_length].strftime("%d-%b-%y"),
        "Actual Price:", round(actual[i][0], 2),
        "Predicted:", round(predicted[i][0], 2)
    )