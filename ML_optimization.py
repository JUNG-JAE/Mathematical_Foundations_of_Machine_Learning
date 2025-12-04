import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====================================
# Data and Design Matrix
# ====================================
x = np.array([0, 1, 2, 3], dtype=float)
y = np.array([1, 2, 5, 10], dtype=float)
n = len(y)

# Design matrix X: rows are [x_i^2, x_i, 1]
X = np.column_stack([x**2, x, np.ones_like(x)])

# ====================================
# Loss Function and Gradient
# L(theta) = (1/n) ||X\theta - y||^2
# \nabla L(theta) = (2/n) X^T (X\theta - y)
# ====================================
def loss(theta):
    residual = X @ theta - y
    return (residual @ residual) / n

def grad(theta):
    residual = X @ theta - y
    return (2.0 / n) * (X.T @ residual)

# ====================================
# Gradient Descent Settings
# ====================================
eta = 0.01
theta = np.zeros(3)     # initial theta = (0, 0, 0)
num_iters = 10

theta_history = [theta.copy()]
loss_history = [loss(theta)]

print("k    theta_1    theta_2    theta_3    L(theta_k)")
print("-" * 55)
print(f"{0:2d}  {theta[0]:8.6f}  {theta[1]:8.6f}  {theta[2]:8.6f}  {loss_history[-1]:10.6f}")

for k in range(1, num_iters + 1):
    g = grad(theta)
    theta = theta - eta * g

    theta_history.append(theta.copy())
    loss_history.append(loss(theta))

    print(f"{k:2d}  {theta[0]:8.6f}  {theta[1]:8.6f}  {theta[2]:8.6f}  {loss_history[-1]:10.6f}")

theta_history = np.array(theta_history)   # shape (num_iters+1, 3)

# ====================================
# Parameter Trajectory (θ1, θ2, θ3)
# ====================================
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

t1 = theta_history[:, 0]
t2 = theta_history[:, 1]
t3 = theta_history[:, 2]

# plot the path with markers
ax.plot(t1, t2, t3, marker="o", linestyle="-")

# label each point with iteration index
for i, (a, b, c) in enumerate(theta_history):
    ax.text(a, b, c, str(i))

ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_zlabel(r"$\theta_3$")

plt.title("Parameter Trajectory")

plt.tight_layout()

# ====================================
# Loss Curve
# ====================================
plt.figure(figsize=(6, 4))
plt.plot(range(len(loss_history)), loss_history, marker="o")
plt.xlabel("Iteration")
plt.ylabel(r"Loss")
plt.title("Loss")
plt.tight_layout()

# ====================================
# Model Curve Evolution
# ====================================
xx = np.linspace(0, 3, 200)

plt.figure(figsize=(7, 5))

for i, theta_i in enumerate(theta_history):
    yy = theta_i[0] * xx**2 + theta_i[1] * xx + theta_i[2]
    plt.plot(xx, yy, color="C0", linewidth=1.2)

    # add iteration label near the end of each curve
    plt.text(xx[-1], yy[-1], f"{i}", fontsize=9)

# plot training data points
plt.scatter(x, y, color="red")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Model Prediction Curve")
plt.tight_layout()

plt.show()
