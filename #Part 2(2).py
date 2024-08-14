#Part 2(2)
import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a 2D array of complex numbers
# Increased resolution and zooming into the Julia set
Y, X = np.mgrid[-1.5:1.5:0.0005, -1.5:1.5:0.0005]  # Adjust the range and resolution as needed

# Load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)  # Important!

# Define the constant c for the Julia set
c = torch.complex(torch.tensor(-0.7), torch.tensor(0.27015))  # 设定julia的复数C complex
c = c.to(device)

# Initialize z and ns
zs = z.clone()  # Updated!
ns = torch.zeros_like(z)

# Transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

# Julia Set computation
for i in range(500):  # Increase iterations for finer detail
    # Compute the new values of z: z^2 + c
    zs_ = zs * zs + c
    # Check if the value has diverged
    not_diverged = torch.abs(zs_) < 4.0
    # Update variables to compute
    ns += not_diverged
    zs = zs_

# Plotting the Julia set
fig = plt.figure(figsize=(16, 10))

def processFractal(a):
    """Display an array of iteration counts as a colorful picture of a fractal."""
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20 * np.cos(a_cyclic),
                          30 + 50 * np.sin(a_cyclic),
                          155 - 80 * np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

# Display the fractal
plt.imshow(processFractal(ns.cpu().numpy()), extent=(-1.5, 1.5, -1.5, 1.5))
plt.tight_layout(pad=0)
plt.show()
