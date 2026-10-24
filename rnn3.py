import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test =x_test/255.0
rnn = Sequential([
    SimpleRNN(64, input_shape=(28, 28)),
    Dense(10, activation='softmax')
])

rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
rnn.fit(x_train, y_train, epochs=3, verbose=1)
rnn_loss, rnn_acc = rnn.evaluate(x_test, y_test, verbose=0)
print('RNN Loss:', rnn_loss)
print('RNN Accuracy:', rnn_acc)
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train_cnn, y_train, epochs=3, verbose=1)
cnn_loss, cnn_acc = cnn.evaluate(x_test_cnn, y_test, verbose=0)
print('CNN Loss:', cnn_loss)
print('CNN Accuracy:', cnn_acc)

print('\nRNN Accuracy:', round(rnn_acc * 100, 2), '%')
print('CNN Accuracy:', round(cnn_acc * 100, 2), '%')
if cnn_acc > rnn_acc:
    print('CNN performs better for image/Character recognition.')
else:
    print('RNN performs better for this dataset.')