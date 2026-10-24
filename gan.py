import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


(x_train, _), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255

x_train = x_train.reshape(-1, 784)


generator = tf.keras.Sequential([

    # Take 100 random numbers
    layers.Input(shape=(100,)),

    
    layers.Dense(128, activation="relu"),

    layers.Dense(784, activation="sigmoid")
])




discriminator = tf.keras.Sequential([

    
    layers.Input(shape=(784,)),

    
    layers.Dense(128, activation="relu"),

    # 1 = Real, 0 = Fake
    layers.Dense(1, activation="sigmoid")
])

g_optimizer = tf.keras.optimizers.Adam(0.0002)

d_optimizer = tf.keras.optimizers.Adam(0.0002)

loss_function = tf.keras.losses.BinaryCrossentropy()

batch_size = 128
epochs = 10

for epoch in range(epochs):

    for start in range(0, len(x_train), batch_size):

        real_images = x_train[
            start:start + batch_size
        ]

        current_batch_size = len(real_images)


        noise = tf.random.normal(
            shape=(current_batch_size, 100)
        )

        with tf.GradientTape() as tape:

            fake_images = generator(noise)

          
            real_output = discriminator(real_images)

            fake_output = discriminator(fake_images)

     
            real_loss = loss_function(
                tf.ones_like(real_output),
                real_output
            )

          
            fake_loss = loss_function(
                tf.zeros_like(fake_output),
                fake_output
            )

            
            d_loss = real_loss + fake_loss

        
        d_gradients = tape.gradient(
            d_loss,
            discriminator.trainable_weights
        )

       
        d_optimizer.apply_gradients(
            zip(
                d_gradients,
                discriminator.trainable_weights
            )
        )

        noise = tf.random.normal(
            shape=(current_batch_size, 100)
        )

        with tf.GradientTape() as tape:

            fake_images = generator(noise)

            fake_output = discriminator(fake_images)

            g_loss = loss_function(
                tf.ones_like(fake_output),
                fake_output
            )

        g_gradients = tape.gradient(
            g_loss,
            generator.trainable_weights
        )

        g_optimizer.apply_gradients(
            zip(
                g_gradients,
                generator.trainable_weights
            )
        )
    print(
        "Epoch:", epoch + 1,
        "Generator Loss:", round(float(g_loss), 4),
        "Discriminator Loss:", round(float(d_loss), 4)
    )

noise = np.random.normal(
    size=(10, 100)
).astype("float32")

new_images = generator.predict(
    noise,
    verbose=0
)

plt.figure(figsize=(10, 2))

for i in range(10):

      plt.subplot(1, 10, i + 1)

      plt.imshow(
          new_images[i].reshape(28, 28),
          cmap="gray"
      )

      plt.axis("off")

      plt.show()