#Part 1(1)
import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# Load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# Transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute 2D Sine function
z = torch.sin(x+2*y)

# Plot the result
plt.imshow(z.cpu().numpy(), extent=(-4, 4, -4, 4))
plt.title("2D Sine Function")
plt.tight_layout()
plt.show()