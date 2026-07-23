import tensorflow as tf
from tensorflow import keras
# Input
X = tf.constant([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=tf.float32)

# Output
Y = tf.constant([
    [0],
    [1],
    [1],
    [0]
], dtype=tf.float32)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train Model
model.fit(X, Y, epochs=500, verbose=0)

# Prediction
print(model.predict(X))