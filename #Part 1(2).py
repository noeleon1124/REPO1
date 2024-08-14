import torch
import numpy as np
import matplotlib.pyplot as plt

# Print PyTorch version
print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

# Load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# Transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian function
gaussian = torch.exp(-(x**2 + y**2) / 2.0)

# Compute 2D sine function
sine = torch.sin(x+2*y)

# Multiply Gaussian by sine function
z = gaussian * sine

# Plot the result
plt.imshow(z.cpu().numpy(), extent=(-4, 4, -4, 4))
plt.title("Gaussian Modulated Sine Wave")
plt.colorbar()
plt.tight_layout()
plt.show()