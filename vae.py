import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)


input_img = layers.Input(shape=(784,))

# Hidden layer
x = layers.Dense(128, activation="relu")(input_img)

z_mean = layers.Dense(2)(x)
z_log_var = layers.Dense(2)(x)

class Sampling(layers.Layer):

    def call(self, inputs):

        mean, log_var = inputs

        
        epsilon = tf.random.normal(shape=tf.shape(mean))

      
        return mean + tf.exp(0.5 * log_var) * epsilon
z = Sampling()([z_mean, z_log_var])


encoder = tf.keras.Model(
    input_img,
    [z_mean, z_log_var, z]
)


latent_input = layers.Input(shape=(2,))

h = layers.Dense(128, activation="relu")(latent_input)

output_img = layers.Dense(
    784,
    activation="sigmoid"
)(h)
decoder = tf.keras.Model(
    latent_input,
    output_img
)


class VAE(tf.keras.Model):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):

        # Get input images
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(data)

            
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        data,
                        reconstruction
                    ),
                    axis=-1
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )


        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        return {"loss": total_loss}


vae = VAE(encoder, decoder)


vae.compile(optimizer="adam")



vae.fit(
    x_train,
    epochs=5,
    batch_size=128
)


random_z = np.random.normal(
    size=(10, 2)
).astype("float32")

new_images = decoder.predict(
    random_z,
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
