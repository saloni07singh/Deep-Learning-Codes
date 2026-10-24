# Restricted Boltzmann Machine using Contrastive Divergence
import numpy as np

# Sample binary data
data = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 1]
])

n_visible = data.shape[1]
n_hidden = 2
learning_rate = 0.1
epochs = 1000

# Initialize weights and biases
np.random.seed(42)

weights = np.random.randn(n_visible, n_hidden) * 0.1
visible_bias = np.zeros(n_visible)
hidden_bias = np.zeros(n_hidden)


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Calculate hidden-unit probabilities
def hidden_probability(v):
    return sigmoid(np.dot(v, weights) + hidden_bias)


# Calculate visible-unit probabilities
def visible_probability(h):
    return sigmoid(np.dot(h, weights.T) + visible_bias)


# Train RBM using Contrastive Divergence (CD-1)
for epoch in range(epochs):

    # Positive phase
    visible = data
    hidden_prob = hidden_probability(visible)

    # Sample hidden units
    hidden_sample = (
        np.random.random(hidden_prob.shape) < hidden_prob
    ).astype(float)

    # Negative phase: reconstruct visible units
    visible_prob = visible_probability(hidden_sample)

    visible_sample = (
        np.random.random(visible_prob.shape) < visible_prob
    ).astype(float)

    hidden_prob_negative = hidden_probability(visible_sample)

    # Update weights and biases
    weights += learning_rate * (
        np.dot(visible.T, hidden_prob)
        - np.dot(visible_sample.T, hidden_prob_negative)
    ) / len(data)

    visible_bias += learning_rate * np.mean(
        visible - visible_sample, axis=0
    )

    hidden_bias += learning_rate * np.mean(
        hidden_prob - hidden_prob_negative, axis=0
    )

    # Calculate reconstruction error
    if (epoch + 1) % 100 == 0:
        error = np.mean((visible - visible_prob) ** 2)
        print("Epoch:", epoch + 1, "Reconstruction Error:", round(error, 4))


# Extract learned features
features = hidden_probability(data)

print("\nLearned Hidden Features:")
print(np.round(features, 3))