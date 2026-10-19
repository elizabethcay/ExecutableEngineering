import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f(X):
    x, y = X
    return np.array([10 * (y - x**2), 1 - x])

def J(X):
    x, y = X
    return np.array([[-20 * x, 10], [-1, 0]])

def residual(X):
    return np.linalg.norm(f(X))

def newton_backtracking(X0, tol=1e-6, max_iter=50):
    X = np.array(X0, dtype=float)
    path = [X.copy()]
    func_calls = 0
    iters = 0

    for _ in range(max_iter):
        res_current = residual(X)
        func_calls += 1

        if res_current < tol:
            break

        dX = np.linalg.solve(J(X), -f(X))
        func_calls += 1

        alpha = 1.0
        while residual(X + alpha * dX) >= res_current and alpha > 1e-4:
            alpha *= 0.5
            func_calls += 1

        X = X + alpha * dX
        path.append(X.copy())
        iters += 1

    return np.array(path), iters, func_calls

def newton_bounded(X0, tol=1e-6, max_iter=50):
    X = np.array(X0, dtype=float)
    path = [X.copy()]
    func_calls = 0
    iters = 0

    for _ in range(max_iter):
        res_current = residual(X)
        func_calls += 1

        if res_current < tol:
            break

        dX = np.linalg.solve(J(X), -f(X))
        func_calls += 1

        def obj_1d(alpha):
            nonlocal func_calls
            func_calls += 1
            return residual(X + alpha * dX)

        res = minimize_scalar(obj_1d, bounds=(0, 1), method='bounded')

        X = X + res.x * dX
        path.append(X.copy())
        iters += 1

    return np.array(path), iters, func_calls

X_init = [-1.5, 3.5]
path_bt, iters_bt, evals_bt = newton_backtracking(X_init)
path_bm, iters_bm, evals_bm = newton_bounded(X_init)

fig, ax = plt.subplots(figsize=(7, 4))

x_grid = np.linspace(-2, 2, 100)
y_grid = np.linspace(-1, 4, 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
Z = np.zeros_like(X_mesh)
for i in range(X_mesh.shape[0]):
    for j in range(X_mesh.shape[1]):
        Z[i,j] = residual([X_mesh[i,j], Y_mesh[i,j]])

contour = ax.contour(X_mesh, Y_mesh, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
ax.plot(path_bt[:,0], path_bt[:,1], 'bo-', label='Backtracking', markersize=5, linewidth=2)
ax.plot(path_bm[:,0], path_bm[:,1], 'rx-', label='Bounded Min', markersize=5, linewidth=2)
ax.plot(1, 1, 'k*', markersize=15, label='Root (1,1)')

ax.set_title("Trajectory on Residual Contour Map", pad=15)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("Backtracking:")
print(f"Newton Steps (Iterations): {iters_bt}")
print(f"Function Evaluations (Computational Cost): {evals_bt}")
print("\nBounded Minimization:")
print(f"Newton Steps (Iterations): {iters_bm}")
print(f"Function Evaluations (Computational Cost): {evals_bm}")