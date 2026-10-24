import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load customer dataset
df = pd.read_csv("customer_data.csv")

print("Original Customer Data")
print(df.head())

# Select the six numeric features
X = df[
    [
        "Annual_Income",
        "Spending_Score",
        "Age",
        "Tenure_Months",
        "Purchase_Frequency",
        "Total_Spend"
    ]
].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Customer Data")
print(X_scaled)

# Build the autoencoder
input_layer = Input(shape=(6,))

encoded = Dense(3, activation="relu")(input_layer)
encoded = Dense(2, activation="relu")(encoded)

decoded = Dense(3, activation="relu")(encoded)
decoded = Dense(6, activation="linear")(decoded)

# Define the autoencoder
autoencoder = Model(input_layer, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

# Train the autoencoder
autoencoder.fit(
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

print("Original Number of Features:", X_scaled.shape[1])
print("Reduced Number of Features:", encoded_data.shape[1])
