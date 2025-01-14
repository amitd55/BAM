import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ContinuousBAM:
    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_features_X = self.inputs.shape[1]
        self.n_features_Y = self.outputs.shape[1]
        self.weights = np.zeros((self.n_features_X, self.n_features_Y))
        self._initialize_weights()

    def _initialize_weights(self):
        for x, y in zip(self.inputs, self.outputs):
            bipolar_x = 2 * np.array(x) - 1
            bipolar_y = 2 * np.array(y) - 1
            self.weights += np.outer(bipolar_x, bipolar_y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_vector=None, output_vector=None, max_iterations=100, tolerance=1e-5):
        x = np.array(input_vector, dtype=float) if input_vector is not None else None
        y = np.array(output_vector, dtype=float) if output_vector is not None else None

        for _ in range(max_iterations):
            if x is not None:
                y_new = self.sigmoid(np.dot(x, self.weights))
                if y is not None and np.allclose(y, y_new, atol=tolerance):
                    break
                y = y_new

            if y is not None:
                x_new = self.sigmoid(np.dot(y, self.weights.T))
                if x is not None and np.allclose(x, x_new, atol=tolerance):
                    break
                x = x_new

        return x, y


# Load the dataset
data = pd.read_csv('Letters_A-Z_BAM_Model_Data.csv')

inputs = data["Input Vector (64-bit)"].apply(lambda x: [int(bit) for bit in str(x)]).tolist()
outputs = data["Output Vector (7-bit)"].apply(lambda x: [int(bit) for bit in str(x)]).tolist()

cbam = ContinuousBAM(inputs, outputs)


# Function to add noise
def add_noise(vector, noise_level):
    noisy_vector = vector.copy()
    num_bits_to_flip = max(1, int(len(vector) * (noise_level / 100)))  # Ensure at least 1 bit flips
    indices_to_flip = np.random.choice(len(vector), num_bits_to_flip, replace=False)
    for idx in indices_to_flip:
        noisy_vector[idx] = 1 - noisy_vector[idx]
    return noisy_vector


# Add printouts and graphs
print("Starting prediction process...\n")

# Noise levels to test
noise_levels = [0, 5, 10, 20]
results = []
predictions = []

for noise_level in noise_levels:
    input_to_output_accuracies = []
    output_to_input_accuracies = []

    print(f"Testing Noise Level: {noise_level}%")
    for i, (original_input, original_output) in enumerate(zip(inputs, outputs)):
        # Add noise to input and predict output
        noisy_input = add_noise(original_input, noise_level)
        _, predicted_output = cbam.predict(input_vector=noisy_input)
        predicted_output_binary = [1 if value >= 0.5 else 0 for value in predicted_output]
        input_to_output_accuracy = np.mean([p == o for p, o in zip(predicted_output_binary, original_output)]) * 100
        input_to_output_accuracies.append(input_to_output_accuracy)

        # Add noise to output and predict input
        noisy_output = add_noise(original_output, noise_level)
        predicted_input, _ = cbam.predict(output_vector=noisy_output)
        predicted_input_binary = [1 if value >= 0.5 else 0 for value in predicted_input]
        output_to_input_accuracy = np.mean([p == o for p, o in zip(predicted_input_binary, original_input)]) * 100
        output_to_input_accuracies.append(output_to_input_accuracy)

        # Print details for all samples
        print(f"  Sample {i+1}:")
        print(f"    Original Input: {original_input}")
        print(f"    Noisy Input: {noisy_input}")
        print(f"    Predicted Output: {predicted_output_binary}")
        print(f"    Original Output: {original_output}")
        print(f"    Noisy Output: {noisy_output}")
        print(f"    Predicted Input: {predicted_input_binary}")

    # Plot bar chart for this noise level
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Input to Output", "Output to Input"],
        [np.mean(input_to_output_accuracies), np.mean(output_to_input_accuracies)],
        color=["blue", "orange"]
    )
    plt.title(f"Accuracy at Noise Level {noise_level}%")
    plt.ylabel("Accuracy (%)")
    plt.ylim(50, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f"  Summary for Noise Level {noise_level}%:")
    print(f"    Mean Input to Output Accuracy: {np.mean(input_to_output_accuracies):.2f}%")
    print(f"    Mean Output to Input Accuracy: {np.mean(output_to_input_accuracies):.2f}%\n")

    results.append({
        "Noise Level (%)": noise_level,
        "Input to Output Accuracy (%)": np.mean(input_to_output_accuracies),
        "Output to Input Accuracy (%)": np.mean(output_to_input_accuracies),
    })

print("Prediction process completed.")
