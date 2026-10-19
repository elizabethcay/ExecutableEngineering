import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import ipywidgets as widgets
from scipy.optimize import minimize

class OptimizerOpen:

    def __init__(self, f, x0, tol=1e-6, max_iter=300):
        self.f = f
        self.x0 = np.asarray(x0, dtype=float)
        self.tol = tol
        self.max_iter = max_iter


        self.powell_steps = self._run_optimization('Powell')
        self.nm_data = self._run_nelder_mead()
        (self.nelder_mead_steps,
         self.nelder_mead_labels,
         self.nelder_mead_simplices,
         self.rejected_points) = self.nm_data

        self._define_plot_domain()

    def _run_optimization(self, method):
        steps = []
        def callback(xk):
            steps.append(xk.copy())
        res = minimize(self.f, self.x0, method=method, callback=callback,
                       options={'maxiter': self.max_iter})
        if not steps or not np.allclose(steps[0], self.x0):
            steps.insert(0, self.x0.copy())
        return [tuple(s) for s in steps]

    def _run_nelder_mead(self):
        α, γ, ρ, σ = 1.0, 2.0, 0.5, 0.5
        x0 = self.x0
        n = len(x0)
        simplex = [x0]
        for i in range(n):
            e = np.zeros(n)
            e[i] = 0.05 if x0[i] == 0 else 0.05 * x0[i]
            simplex.append(x0 + e)
        simplex = np.array(simplex)
        fvals = np.array([self.f(v) for v in simplex])

        steps = [x0.copy()]
        labels = ["Initial simplex"]
        simplices = [simplex.copy()]
        rejected_points = [None]

        for _ in range(self.max_iter):
            order = np.argsort(fvals)
            simplex, fvals = simplex[order], fvals[order]
            centroid = np.mean(simplex[:-1], axis=0)

            xr = centroid + α * (centroid - simplex[-1])
            fr = self.f(xr)

            label = None
            rejected_point = None

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
                label = "Reflection accepted"
            elif fr < fvals[0]:
                xe = centroid + γ * (xr - centroid)
                fe = self.f(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                    label = "Reflection accepted → Expansion accepted"
                    rejected_point = xr
                else:
                    simplex[-1], fvals[-1] = xr, fr
                    label = "Expansion rejected → Reflection accepted"
                    rejected_point = xe
            else:
                xc = centroid + ρ * (simplex[-1] - centroid)
                fc = self.f(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                    label = "Reflection rejected → Contraction accepted"
                    rejected_point = xr
                else:
                    rejected_point = xc
                    best = simplex[0]
                    for j in range(1, len(simplex)):
                        simplex[j] = best + σ * (simplex[j] - best)
                        fvals[j] = self.f(simplex[j])
                    label = "Contraction rejected → Shrinkage accepted"

            steps.append(centroid.copy())
            labels.append(label)
            simplices.append(simplex.copy())
            rejected_points.append(rejected_point.copy() if rejected_point is not None else None)

            if np.std(fvals) < self.tol:
                break

        return (
            [tuple(s) for s in steps],
            labels,
            simplices,
            rejected_points
        )

    def _define_plot_domain(self):
        all_points = np.array(self.powell_steps + self.nelder_mead_steps)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        dx, dy = (x_max - x_min) * 0.3, (y_max - y_min) * 0.3
        x_min, x_max = x_min - dx, x_max + dx
        y_min, y_max = y_min - dy, y_max + dy
        self.x = np.linspace(x_min, x_max, 150)
        self.y = np.linspace(y_min, y_max, 150)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = np.array([[self.f([xi, yi]) for xi in self.x] for yi in self.y])

    def _get_color_for_rejection(self, label):
        if "Reflection rejected" in label:
            return "blue"
        elif "Expansion rejected" in label:
            return "green"
        elif "Contraction rejected" in label:
            return "orange"
        elif "Shrinkage accepted" in label:
            return "purple"
        return None

    def _animate_optimization(self, steps, method_name, interval_ms=500,
                             labels=None, simplices=None, rejected_points=None, show_vectors=False):
        if not steps:
            return None
        fig, ax = plt.subplots(figsize=(8,8))
        contour = ax.contourf(self.X, self.Y, self.Z, levels=50, cmap='viridis')
        ax.contour(self.X, self.Y, self.Z, levels=15, colors='k', alpha=0.3, linewidths=0.5)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("f(x, y)", rotation=270, labelpad=15)

        ax.plot(self.x0[0], self.x0[1], 'r*', markersize=10, label='Initial Guess')


        path_line, = ax.plot([], [], 'r-', lw=1.5, alpha=0.7)
        current_dot, = ax.plot([], [], 'ro', markersize=6, label='Current Point')

        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])

        title_text = ax.text(0.5, 1.03, '', transform=ax.transAxes,

                             ha='center', fontsize=12)

        indicator_box = ax.annotate(
            '',
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            textcoords='axes fraction',
            ha='left',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="black")
        )

        ax.legend(loc='lower left')
        simplex_lines, rejected_markers = [], []
        quivers = []

        def update(i):
            for ln in simplex_lines:
                ln.remove()
            for m in rejected_markers:
                m.remove()
            simplex_lines.clear()
            rejected_markers.clear()

            px, py = [s[0] for s in steps[:i+1]], [s[1] for s in steps[:i+1]]
            path_line.set_data(px, py)
            cx, cy = steps[i]
            current_dot.set_data([cx], [cy])

            if simplices is not None and i < len(simplices):
                tri = simplices[i]
                tri_closed = np.vstack([tri, tri[0]])
                ln, = ax.plot(tri_closed[:,0], tri_closed[:,1], 'w-', lw=2, alpha=0.9)
                simplex_lines.append(ln)


            for q in quivers:
                q.remove()
            quivers.clear()

            if show_vectors and i > 0:
                x_prev, y_prev = steps[i-1]
                dx, dy = current_x - x_prev, current_y - y_prev
                q = ax.quiver(x_prev, y_prev, dx, dy, angles='xy', scale_units='xy',
                              scale=1, color='red', width=0.005)
                quivers.append(q)

            label_text = labels[i] if labels and i < len(labels) else ""
            title_text.set_text(
                f"{method_name} Step {i+1}/{len(steps)}\n"
                f"({cx:.4f}, {cy:.4f})  f={self.f([cx, cy]):.4f}"
            )
            if method_name == "Nelder–Mead":
                indicator_box.set_text(
                    f"Action: {label_text}\n"
                )

            return path_line, current_dot, title_text, indicator_box, *simplex_lines, *rejected_markers

        anim = FuncAnimation(fig, update, frames=len(steps),
                             interval=interval_ms, blit=True, repeat=False)
        plt.close(fig)
        return HTML(anim.to_jshtml())

    def show_toggle_open(self, interval_ms=500):
        powell_anim = self._animate_optimization(
            self.powell_steps, "Powell's Method", interval_ms, show_vectors=True)
        nm_anim = self._animate_optimization(
            self.nelder_mead_steps, "Nelder–Mead", interval_ms,
            labels=self.nelder_mead_labels,
            simplices=self.nelder_mead_simplices,
            rejected_points=self.rejected_points)
        powell_tab = widgets.HTML(value=powell_anim.data)
        nm_tab = widgets.HTML(value=nm_anim.data)
        tab = widgets.Tab()
        tab.children = [powell_tab, nm_tab]
        tab.set_title(0, "Powell's Method")
        tab.set_title(1, "Nelder–Mead Method")
        display(tab)


def plot_optimization_path_3d(func, path, x_range=None, y_range=None, displacement_factor=0.8, title="3D Optimization Surface & Path"):
    """
    Plots an interactive 3D surface with floor contour shadow and the optimization path.
    
    Args:
        func: Callable taking a 2-element sequence [x1, x2] and returning a scalar.
        path: Array-like of shape (N, 2) containing the sequence of iterate points.
        x_range: Tuple of (x_min, x_max). If None, inferred with margin from path.
        y_range: Tuple of (y_min, y_max). If None, inferred with margin from path.
        displacement_factor: Fraction of z-span to displace the floor below the surface minimum.
        title: Plot title.
    """
    import numpy as np
    import plotly.graph_objects as go

    pts = np.asarray(path, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("path must be of shape (N, 2)")

    if x_range is None:
        dx = max(np.ptp(pts[:, 0]) * 0.35, 1.0)
        x_range = (np.min(pts[:, 0]) - dx, np.max(pts[:, 0]) + dx)
    if y_range is None:
        dy = max(np.ptp(pts[:, 1]) * 0.35, 1.0)
        y_range = (np.min(pts[:, 1]) - dy, np.max(pts[:, 1]) + dy)

    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    z_pts = np.array([float(func(pt)) for pt in pts])
    z_min_surf = float(np.min(Z))
    z_max_surf = float(np.max(Z))
    z_span = z_max_surf - z_min_surf if z_max_surf > z_min_surf else 1.0
    z_gap = displacement_factor * z_span
    z_floor = z_min_surf - z_gap
    steps = np.arange(len(pts))

    fig = go.Figure()

    # 3D surface with bottom-projected contours (shadow)
    fig.add_trace(go.Surface(
        x=x_vals, y=y_vals, z=Z,
        colorscale="Viridis",
        opacity=0.85,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        ),
        colorbar=dict(title="f(x1, x2)", len=0.75, x=1.05),
        name="Surface"
    ))

    # 3D path line along the surface
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=z_pts,
        mode="lines",
        line=dict(color="red", width=5),
        name="Path (3D)"
    ))

    # 3D iterate markers colored by step number
    hover_texts = [
        f"Step {k}<br>x1 = {pt[0]:.4f}<br>x2 = {pt[1]:.4f}<br>f = {zk:.4e}"
        for k, (pt, zk) in enumerate(zip(pts, z_pts))
    ]

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=z_pts,
        mode="markers",
        marker=dict(
            size=5,
            color=steps,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Iteration", len=0.75, x=-0.08)
        ),
        name="Iterates",
        text=hover_texts,
        hoverinfo="text"
    ))

    # Projected 2D shadow path on the floor plane z = z_floor
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=[z_floor] * len(pts),
        mode="lines+markers",
        line=dict(color="rgba(40, 40, 40, 0.6)", width=3, dash="dot"),
        marker=dict(size=3, color="black"),
        name="Floor Projection"
    ))

    # Start and End points
    fig.add_trace(go.Scatter3d(
        x=[pts[0, 0]], y=[pts[0, 1]], z=[z_pts[0]],
        mode="markers",
        marker=dict(size=8, color="cyan", symbol="circle"),
        name=f"Start ({pts[0, 0]:.2f}, {pts[0, 1]:.2f})"
    ))

    fig.add_trace(go.Scatter3d(
        x=[pts[-1, 0]], y=[pts[-1, 1]], z=[z_pts[-1]],
        mode="markers",
        marker=dict(size=10, color="crimson", symbol="diamond"),
        name=f"End ({pts[-1, 0]:.4f}, {pts[-1, 1]:.4f})"
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="f(x1, x2)",
            zaxis=dict(range=[z_floor, z_max_surf + 0.05 * z_span]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.9),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))
        ),
        width=800,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    fig.show()


# Backward-compatible alias
plot_optimization_path_2d = plot_optimization_path_3d


