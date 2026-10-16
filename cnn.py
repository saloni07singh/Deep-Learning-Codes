import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)


# ============================================================
# 2. PREPROCESSING
# ============================================================

# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension
# 28 x 28  ->  28 x 28 x 1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ============================================================
# 3. DISPLAY SAMPLE IMAGE
# ============================================================

plt.imshow(x_train[0].squeeze(), cmap="gray")
plt.title("Sample MNIST Image")
plt.xlabel(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()


# ============================================================
# 4. BUILD CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    # Convolution Layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # Pooling Layer
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second Convolution Layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Second Pooling Layer
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convert feature maps into one-dimensional vector
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output Layer
    # 10 classes: 0-9
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 5. DISPLAY MODEL STRUCTURE
# ============================================================

model.summary()


# ============================================================
# 6. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)


# ============================================================
# 8. TEST / EVALUATE MODEL
# ============================================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 9. PLOT ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 10. PLOT LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 11. PREDICT A TEST IMAGE
# ============================================================

index = 0

prediction = model.predict(
    x_test[index:index + 1],
    verbose=0
)

predicted_class = prediction.argmax()

print("\nPredicted Digit:", predicted_class)
print("Actual Digit:", y_test[index])


# ============================================================
# 12. DISPLAY IMAGE WITH PREDICTION
# ============================================================

plt.figure(figsize=(4, 4))

plt.imshow(
    x_test[index].squeeze(),
    cmap="gray"
)

plt.title(
    f"Predicted: {predicted_class} | Actual: {y_test[index]}"
)

plt.axis("off")
plt.show()