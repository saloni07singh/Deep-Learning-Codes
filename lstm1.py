import numpy as np

import tensorflow as tf
from PIL import Image

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add
from tensorflow.keras.models import Model


# Sample captions
captions = [
    "<start> a dog is sitting <end>",
    "<start> a dog is lying down <end>",
    "<start> a brown dog <end>",
    "<start> a furry dog <end>"
]


# Load image from the same folder as this Python file
image_path = "dog.jpg"

image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))
image = np.array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)


# Extract image features using MobileNetV2
cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

image_features = cnn.predict(image, verbose=0)

print("Image features extracted successfully.")


# Tokenize captions
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(captions)

sequences = tokenizer.texts_to_sequences(captions)

max_length = max(len(seq) for seq in sequences)
vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary size:", vocab_size)


# Prepare input and output sequences
X_text = []
y_text = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        X_text.append(sequence[:i])
        y_text.append(sequence[i])

X_text = pad_sequences(
    X_text,
    maxlen=max_length,
    padding="post"
)

y_text = np.array(y_text)


# Build LSTM model
text_input = Input(shape=(max_length,))

text_embedding = Embedding(
    vocab_size, 128
)(text_input)

lstm_output = LSTM(128)(text_embedding)

image_input = Input(shape=(1280,))

image_dense = Dense(128, activation="relu")(image_input)

combined = Add()([image_dense, lstm_output])

output = Dense(
    vocab_size,
    activation="softmax"
)(combined)

model = Model(
    inputs=[image_input, text_input],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)


# Train the model
image_training = np.repeat(
    image_features,
    len(X_text),
    axis=0
)

print("\nTraining LSTM...")

model.fit(
    [image_training, X_text],
    y_text,
    epochs=100,
    batch_size=8,
    verbose=1
)

print("Training completed.")


# Generate caption
current_sequence = [tokenizer.word_index["<start>"]]

print("\nGenerating caption...")

for i in range(max_length - 1):

    input_sequence = pad_sequences(
        [current_sequence],
        maxlen=max_length,
        padding="post"
    )

    prediction = model.predict(
        [image_features, input_sequence],
        verbose=0
    )

    next_word = np.argmax(prediction[0])

    if next_word == tokenizer.word_index["<end>"]:
        break

    current_sequence.append(next_word)


# Convert word indexes into a caption
index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}

predicted_words = [
    index_to_word[index]
    for index in current_sequence
    if index in index_to_word
    and index not in [
        tokenizer.word_index["<start>"],
        tokenizer.word_index["<end>"]
    ]
]

print("\nPredicted Caption:")
print(" ".join(predicted_words))