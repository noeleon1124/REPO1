#Part 3
import torch
import matplotlib.pyplot as plt

def draw_cross(fractal, center, size):
    """在给定位置和大小的图像上绘制十字架"""
    cx, cy = center
    half_size = size // 2

    # 绘制水平和垂直的线条
    fractal[cy - half_size:cy + half_size + 1, cx - 1:cx + 2] = 1
    fractal[cy - 1:cy + 2, cx - half_size:cx + half_size + 1] = 1

def draw_fractal(fractal, center, size, depth):
    """递归绘制希腊十字分形"""
    if depth == 0 or size < 3:
        return

    # 绘制当前层的十字架
    draw_cross(fractal, center, size)

    # 计算新的中心和大小
    new_size = size // 3

    # 子十字架的位置
    offsets = [
        (-new_size, -new_size), (new_size, -new_size),
        (-new_size, new_size), (new_size, new_size)
    ]

    # 递归绘制每个子十字架
    for offset in offsets:
        new_center = (center[0] + offset[0], center[1] + offset[1])
        draw_fractal(fractal, new_center, new_size, depth - 1)

# 定义图像大小和递归深度
size = 243  # 图像大小，建议使用 3 的幂
depth = 4   # 递归深度

# 创建一个空的图像
fractal = torch.zeros((size, size), dtype=torch.float32)

# 开始绘制希腊十字分形
center = (size // 2, size // 2)
draw_fractal(fractal, center, size, depth)

# 显示图像
plt.imshow(fractal.numpy(), cmap='gray')
plt.axis('off')
plt.show()