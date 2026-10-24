import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load customer dataset
df = pd.read_csv("customer_data.csv")

print("Original Customer Data")
print(df.head())

# Select the six required numeric columns
df = df[
    [
        "Annual_Income",
        "Spending_Score",
        "Age",
        "Tenure_Months",
        "Purchase_Frequency",
        "Total_Spend"
    ]
]

# Convert columns to numeric and remove missing values
df = df.apply(pd.to_numeric, errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

X = df.values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Customer Data")
print(X_scaled)

# Build the autoencoder
input_layer = Input(shape=(6,))

encoded = Dense(3, activation="relu")(input_layer)
encoded = Dense(2, activation="linear")(encoded)

decoded = Dense(3, activation="relu")(encoded)
decoded = Dense(6, activation="linear")(decoded)

# Define and compile the autoencoder
autoencoder = Model(input_layer, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

# Train the autoencoder
history = autoencoder.fit(
    X_scaled,
    X_scaled,
    epochs=100,
    batch_size=4,
    verbose=0
)

# Create the encoder
encoder = Model(input_layer, encoded)

# Generate reduced features
encoded_data = encoder.predict(X_scaled)

print("\nEncoded Features:")
print(encoded_data)

print("\nOriginal Number of Features:", X_scaled.shape[1])
print("Reduced Number of Features:", encoded_data.shape[1])
print("Number of Customer Records:", encoded_data.shape[0])
print("Final Training Loss:", history.history["loss"][-1])
