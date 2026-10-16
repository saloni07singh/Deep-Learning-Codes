import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


df = pd.read_csv("/content/Dry_Bean_Dataset.csv")
df.head()
df['Class'].value_counts()


encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])

X = df.drop("Class", axis=1)
X.head()
y = df["Class"]
y.tail(10)

# Step 4: Scale Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, batch_size=32)
# Step 9: Predict
prediction = model.predict(X_test)
prediction = prediction.argmax(axis=1)

accuracy = accuracy_score(y_test, prediction)
print("Accuracy=", accuracy)

sample = X_test[0:1]
predicted_class = model.predict(sample).argmax(axis=1)
print("Predicted Class:", encoder.inverse_transform(predicted_class))
