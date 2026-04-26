"""Template for the multi class Perceptron digit classifier.

Implement the perceptron training and prediction logic yourself. Do
not call sklearn, torch, or any other library's perceptron.

Required public API (fixed for auto grading):
  * class `PerceptronDigitsClassifier` with methods `train`, `predict`,
    `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)` which runs
    the full train/test pipeline and prints results in the standard
    format below.

Usage:
    python3 q1a_perceptron_digits.py <training_percent>
    e.g.  python3 q1a_perceptron_digits.py 50
"""

import sys
import time
import numpy as np

from util_digits import load_digits


class PerceptronDigitsClassifier:
    """Multi class Perceptron for the 10 digit classes.

    Implementation notes:
      * Keep one weight vector (and bias) per class, 10 in total.
      * For image x, predict `argmax_y (w_y . x + b_y)`.
      * On a misclassification, reinforce the true class weights and
        penalize the mispredicted class weights.
      * Iterate over the training set `max_iterations` times.
    """

    def __init__(self, num_classes: int = 10, image_shape=(28, 28),
                 max_iterations: int = 3):
        """Initialise weights and biases.

        `num_classes` is the number of output classes (10 for digits).
        `image_shape` is (rows, cols) for each input image; it sizes
        the weight tensor. `max_iterations` is the number of full
        passes over the training set during `train`.
        """
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.max_iterations = max_iterations
        self.weights = np.zeros((num_classes, image_shape[0] * image_shape[1]),dtype=np.float32)
        self.biases = np.zeros(num_classes, dtype=np.float32)

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Fit the perceptron on training data.

        `training_images` has shape (N, 28, 28). `training_labels` has
        shape (N,) with values in {0..9}.
        """
        flat_images = training_images.reshape(training_images.shape[0], -1).astype(
            np.float32, copy=False
        )
        labels = training_labels.astype(np.int64, copy=False)

        for _ in range(self.max_iterations):
            for x, true_label in zip(flat_images, labels):
                predicted = int(np.argmax(self.weights @ x + self.biases))
                if predicted != true_label:
                    self.weights[true_label] += x
                    self.weights[predicted] -= x
                    self.biases[true_label] += 1.0
                    self.biases[predicted] -= 1.0

    def predict(self, image: np.ndarray) -> int:
        """Predict a label in {0..9} for a single 28x28 image."""
        x = image.reshape(-1).astype(np.float32, copy=False)
        return int(np.argmax(self.weights @ x + self.biases))

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy in [0, 1] over a batch."""
        correct = 0
        for image, label in zip(images, labels):
            if self.predict(image) == label:
                correct += 1
        return correct / len(labels)


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the digit perceptron.

    Protocol from the project handout:
      * Sample `training_percent` percent of the training data
        uniformly at random, `num_iterations` times.
      * For each sample, train a fresh classifier and evaluate it on
        the full test set.
      * Report mean and std of prediction error and mean training time.

    Returns a dict with keys `training_percent`, `mean_train_time`,
    `mean_error`, `std_error`, `mean_accuracy`, `std_accuracy`.
    """
    training_images, training_labels = load_digits("training")
    test_images, test_labels = load_digits("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        clf = PerceptronDigitsClassifier()
        start = time.time()
        clf.train(x_sample, y_sample)
        train_times[i] = time.time() - start

        accuracies[i] = clf.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }

    print(f"\n=== Perceptron | Digits | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
