import numpy as np
import matplotlib.pyplot as plt

class GaussianWeight():
    def __init__(self, height, width, sigma=1.0):
        self.sigma = sigma
        self.height = height
        self.width = width
        self.weights = self.generate_weights()

    def generate_weights(self):
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        d = np.sqrt(xx * xx + yy * yy)
        weights = np.exp(-(d ** 2) / (2.0 * self.sigma ** 2))
        return weights

def mse_loss_with_gaussian_weighting(y_true, y_pred, sigma=1.0):
    height, width, channels = y_true.shape
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Generate Gaussian weights
    weights = GaussianWeight(height, width, sigma)

    # Apply weights to the squared error
    squared_error = (y_true_flat - y_pred_flat) ** 2
    weighted_error = squared_error * weights.flatten()

    # Calculate the mean of the weighted error
    loss = np.mean(weighted_error)
    return loss

# Generate Gaussian weights
x = np.linspace(-1, 1, 500)
y = np.linspace(-1, 1, 500)
xx, yy = np.meshgrid(x, y)
d = np.sqrt(xx * xx + yy * yy)
weights = np.exp(-(d ** 2) / (2.0 * 1 ** 2))
print(weights.shape)

# Plot the weights
plt.imshow(weights, cmap='rainbow')
plt.show()