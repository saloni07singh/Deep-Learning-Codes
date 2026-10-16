import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
df = pd.read_csv("customer_data.csv")

print("Original Customer Data")
df.head()

X =df.values
X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)


input_layer = Input(shape=(6,))
encoded = Dense(3, activation='relu')(input_layer)
encoded = Dense(2, activation='relu')(encoded)
decoded = Dense(3, activation='relu')(encoded)
decoded = Dense(6, activation='linear')(decoded)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

autoencoder.fit(
    X_scaled,
    X_scaled,
    epochs=100,
    batch_size= 4,
    verbose = 0
)

encoder = Model(input_layer, encoded)
print("\nEncoded Features:")
print(encoded_data)
print("Original Number of Features:", X_scaled.shape[1])
print("Reduced Number of Features:", encoded_data.shape[1])