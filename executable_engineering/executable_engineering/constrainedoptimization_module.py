import numpy as np
import plotly.graph_objects as go


def plot_constrained_optimum_3d(
    func,
    optimum,
    constraint_curve=None,
    x_range=(-1.5, 1.5),
    y_range=(-1.5, 1.5),
    x0=None,
    path=None,
    displacement_factor=0.8,
    title="Constrained Optimization Surface",
):
    """
    Plots an interactive 3D objective surface, constraint boundary, iterates, and the constrained optimum.

    Args:
        func: Callable taking a 2-element sequence [x, y] and returning a scalar f(x, y).
        optimum: Sequence [x*, y*] representing the constrained optimum.
        constraint_curve: Optional array-like of shape (M, 2) defining (x, y) coordinates of the constraint boundary.
        x_range: Tuple of (x_min, x_max).
        y_range: Tuple of (y_min, y_max).
        x0: Optional sequence [x0, y0] representing the initial starting guess.
        path: Optional array-like of shape (N, 2) representing intermediate iterates.
        displacement_factor: Fraction of z-span to displace the floor plane below the surface minimum.
        title: Plot title.
    """
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    z_min_surf = float(np.min(Z))
    z_max_surf = float(np.max(Z))
    z_span = z_max_surf - z_min_surf if z_max_surf > z_min_surf else 1.0
    z_floor = z_min_surf - displacement_factor * z_span

    fig = go.Figure()

    # 3D surface with bottom-projected contours (shadow)
    fig.add_trace(
        go.Surface(
            x=x_vals,
            y=y_vals,
            z=Z,
            colorscale="Viridis",
            opacity=0.8,
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True,
                )
            ),
            colorbar=dict(title="f(x, y)", len=0.75, x=1.05),
            name="Surface",
        )
    )

    # Constraint curve (both along surface and on floor plane)
    if constraint_curve is not None:
        c_pts = np.asarray(constraint_curve, dtype=float)
        cz_surf = np.array([float(func(pt)) for pt in c_pts])

        # Constraint along surface
        fig.add_trace(
            go.Scatter3d(
                x=c_pts[:, 0],
                y=c_pts[:, 1],
                z=cz_surf,
                mode="lines",
                line=dict(color="orange", width=6),
                name="Constraint on Surface",
            )
        )

        # Constraint projected on floor
        fig.add_trace(
            go.Scatter3d(
                x=c_pts[:, 0],
                y=c_pts[:, 1],
                z=[z_floor] * len(c_pts),
                mode="lines",
                line=dict(color="rgba(255, 140, 0, 0.7)", width=4, dash="dash"),
                name="Constraint on Floor",
            )
        )

    # Optional optimization path
    if path is not None:
        pts = np.asarray(path, dtype=float)
        if pts.ndim == 2 and len(pts) > 0:
            pz = np.array([float(func(pt)) for pt in pts])
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pz,
                    mode="lines+markers",
                    line=dict(color="red", width=4),
                    marker=dict(size=4, color="crimson"),
                    name="Solver Path",
                )
            )

    # Initial guess
    if x0 is not None:
        pt0 = np.asarray(x0, dtype=float)
        z0 = float(func(pt0))
        fig.add_trace(
            go.Scatter3d(
                x=[pt0[0]],
                y=[pt0[1]],
                z=[z0],
                mode="markers",
                marker=dict(size=8, color="cyan", symbol="circle"),
                name=f"Initial Guess ({pt0[0]:.2f}, {pt0[1]:.2f})",
            )
        )

    # Optimum point
    opt = np.asarray(optimum, dtype=float)
    z_opt = float(func(opt))
    fig.add_trace(
        go.Scatter3d(
            x=[opt[0]],
            y=[opt[1]],
            z=[z_opt],
            mode="markers",
            marker=dict(size=10, color="crimson", symbol="diamond"),
            name=f"Optimum ({opt[0]:.3f}, {opt[1]:.3f})",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)",
            zaxis=dict(range=[z_floor, z_max_surf + 0.05 * z_span]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.9),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3)),
        ),
        width=750,
        height=550,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
        ),
    )
    fig.show()
