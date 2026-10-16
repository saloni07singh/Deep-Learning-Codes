import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
autoencoder = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation="sigmoid")
])
autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)
autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=256,
    validation_data=(x_test, x_test)
)
reconstructed = autoencoder.predict(x_test)
plt.figure(figsize=(10, 4))

for i in range(5):
  plt.subplot(2, 5, i + 1)

  plt.imshow(x_test[i].reshape(28, 28), cmap="gray")

  plt.axis("off")

  plt.title("Original")

  #reconstructed image

  plt.subplot(2, 5, i + 6)

  plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")

  plt.axis("off")

  plt.title("Reconstructed")

plt.show()
