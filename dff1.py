import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/train.csv")
df.head()


df = df.drop("id", axis=1)

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
df["Vehicle_Age"] = encoder.fit_transform(df["Vehicle_Age"])
df["Vehicle_Damage"] = encoder.fit_transform(df["Vehicle_Damage"])


X = df.drop("Response", axis=1)
y = df["Response"]

scalar = StandardScaler()
X = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn import model_selection

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(32,activation="relu"))

model.add(tf.keras.layers.Dense(16,activation="relu"))

model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


model.compile(optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32
)

X_test

prediction = model.predict(X_test)
prediction = (prediction > 0.5).astype(int)
accuracy = accuracy_score(y_test, prediction)
print("Accuracy:",accuracy)
