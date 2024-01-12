import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define Kuramoto model function
def kuramoto_model(t, theta, K, omega, A):
    N = theta.shape[0]
    dtheta_dt = omega - K * np.sum(A * np.sin(theta[:, np.newaxis] - theta), axis=1)
    return dtheta_dt

# Hexagonal grid generation
def hexagonal_grid(size):
    A = np.zeros((size**2, size**2))
    for i in range(size):
        for j in range(size):
            idx = i * size + j
            if j > 0:
                A[idx, idx-1] = 1
            if j < size - 1:
                A[idx, idx+1] = 1
            if i > 0:
                A[idx, idx-size] = 1
                if j < size - 1:
                    A[idx, idx-size+1] = 1
            if i < size - 1:
                A[idx, idx+size] = 1
                if j > 0:
                    A[idx, idx+size-1] = 1
    return A

# Parameters
size = 10
N = size**2
K = 10
t_max = 100
dt = 0.01

# Initialize
omega = np.random.uniform(-1, 1, N)
theta_0 = np.random.uniform(0, 2 * np.pi, N)
A = hexagonal_grid(size)

# Integration using the Euler method
t_vals = np.arange(0, t_max, dt)
theta_vals = np.zeros((N, t_vals.size))
theta_vals[:, 0] = theta_0

for i in range(1, t_vals.size):
    theta_vals[:, i] = theta_vals[:, i - 1] + dt * kuramoto_model(t_vals[i - 1], theta_vals[:, i - 1], K, omega, A)

# Visualization
x_coords = np.zeros(N)
y_coords = np.zeros(N)
for i in range(size):
    for j in range(size):
        idx = i * size + j
        x_coords[idx] = j + 0.5 * (i % 2)
        y_coords[idx] = np.sqrt(3) * i / 2

fig, ax = plt.subplots()
scat = ax.scatter(x_coords, y_coords, c=theta_vals[:, 0], cmap='hsv', vmin=0, vmax=2 * np.pi)

def update(frame):
    scat.set_array(theta_vals[:, frame])
    return scat,

ani = FuncAnimation(fig, update, frames=t_vals.size, interval=1000 * dt, blit=True)
ax.set_aspect('equal')
plt.colorbar(scat)
plt.show()