import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add
from tensorflow.keras.models import Model


captions = [
    "a dog is sitting",
    "a dog is lying down",
    "a brown dog",
    "a furry dog"
]

captions = [
    "<start> " + caption + " <end>"
    for caption in captions
]
image_path = "/content/dog.jpeg"
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))
image = np.array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)
image_features = cnn.predict(image, verbose=0)
print("Image features extracted successfully.")
print("Feature shape:", image_features.shape)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
sequences = tokenizer.texts_to_sequences(captions)
max_length = max(len(sequence) for sequence in sequences)
print("Vocabulary size:", len(tokenizer.word_index) + 1)
print("Maximum caption length:", max_length)

X_text = []
y_text = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        input_sequence = sequence[:i]
        output_word = sequence[i]
        X_text.append(input_sequence)
        y_text.append(output_word)

X_text = pad_sequences(
    X_text,
    maxlen=max_length,
    padding="post"
)
y_text = np.array(y_text)


vocab_size = len(tokenizer.word_index) + 1
text_input = Input(shape=(max_length,))
text_embedding = Embedding(
    vocab_size,
    128,
    input_length=max_length
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
    verbose=0
)

print("Training completed.")

if not hasattr(tokenizer, 'word_index') or '<start>' not in tokenizer.word_index:
    print("Re-initializing and fitting tokenizer for caption generation.")

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(captions)

start_token = tokenizer.word_index["<start>"]

current_sequence = [start_token]

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

index_to_word = {
    value: key
    for key, value in tokenizer.word_index.items()
}

predicted_words = [
    index_to_word.get(index, "")
    for index in current_sequence
]

predicted_words = [
    word for word in predicted_words
    if word not in ["<start>", "<end>"]
]

caption = " ".join(predicted_words)

print("\nPredicted Caption:")
print(caption)