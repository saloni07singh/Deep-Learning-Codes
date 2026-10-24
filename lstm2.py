import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#training sequence
data = np.array([10,20,30,40,50,60,70,80,90,100])
print(data)
#create input-output pairs

X = []
Y = []
for i in range(len(data) -3):
  X.append(data[i:i+3])
  Y.append(data[i+3])

X = np.array(X)
Y = np.array(Y)
#reshape fpr lstm

X = X.reshape((X.shape[0], X.shape[1], 1))

#create lstm model

model = Sequential([
    LSTM(50, activation='relu', input_shape=(3,1)),
    Dense(1)

])



#compile model
model.compile(
    optimizer='adam',
    loss='mse'
)
#train model

model.fit(X, Y, epochs=200, verbose=0)
#predict next value

test = np.array([80, 90, 100])
test = test.reshape((1 ,3 ,1))
prediction = model.predict(test)
print("predicted next value:", prediction[0][0])