import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression using SGD
model_sgd = tf.keras.Sequential([
    tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        input_shape=(X_train.shape[1],)
    )
])

model_sgd.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_sgd.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=0
)

pred_sgd = (model_sgd.predict(X_test) > 0.5).astype(int)

acc_sgd = accuracy_score(y_test, pred_sgd)

print("Accuracy using SGD :", acc_sgd)

# Logistic Regression using Adam
model_adam = tf.keras.Sequential([
    tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        input_shape=(X_train.shape[1],)
    )
])

model_adam.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_adam.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=0
)

pred_adam = (model_adam.predict(X_test) > 0.5).astype(int)

acc_adam = accuracy_score(y_test, pred_adam)

print("Accuracy using Adam :", acc_adam)

# Linear Regression using Neural Network
linear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        1,
        activation='linear',
        input_shape=(X_train.shape[1],)
    )
])

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

linear_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=0
)

pred = linear_model.predict(X_test)

print("\nSample Linear Regression Predictions:")
print(pred[:10])