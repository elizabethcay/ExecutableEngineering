import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import ipywidgets as widgets

class OptimizerClosed:
    def __init__(self, f, a=None, b=None, tol=1e-6, max_iter=30):
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.gold_steps = self._run_golden_section()
        self.brent_steps = self._run_brent()

    def _run_golden_section(self):
        if self.a is None or self.b is None:
            return []

        a, b = self.a, self.b
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        steps = []

        for _ in range(self.max_iter):
            steps.append((a, b, c, d))

            if abs(b - a) < self.tol:
                break

            if self.f(c) < self.f(d):
                b = d
                d = c
                c = b - gr * (b - a)
            else:
                a = c
                c = d
                d = a + gr * (b - a)
        return steps


    def _run_brent(self):

        if self.a is None or self.b is None:
            return []

        a, b = self.a, self.b
        C = (3.0 - np.sqrt(5.0)) / 2.0

        x = w = v = a + C * (b - a)
        fx = fw = fv = self.f(x)

        d = e = b - a
        steps = []

        for iteration in range(self.max_iter):
            m = 0.5 * (a + b)
            tol1 = self.tol * abs(x) + 1e-12
            tol2 = 2.0 * tol1

            if abs(x - m) <= tol2 - 0.5 * (b - a):
                break

            parabolic_accepted = False
            method = "golden"
            g = e

            if abs(g) > tol1:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)

                if q != 0:
                    if q > 0:
                        p = -p
                    q = abs(q)

                    p_over_q = p / q
                    u_candidate = x + p_over_q

                    cond1 = (a + tol2) <= u_candidate <= (b - tol2)
                    cond2 = abs(p_over_q) < abs(0.5 * g)

                    if cond1 and cond2:
                        u = u_candidate
                        method = "parabolic"
                        parabolic_accepted = True

            if not parabolic_accepted:
                if x < m:
                    e = b - x
                else:
                    e = a - x
                d = C * e
                u = x + d
                method = "golden"

            if abs(u - x) < tol1:
                u = x + np.sign(u - x) * tol1

            fu = self.f(u)

            steps.append((a, b, x, w, v, u, method))

            if fu <= fx:
                if u >= x:
                    a = x
                else:
                    b = x
                v, fv = w, fw
                w, fw = x, fx
                x, fx = u, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if (fu <= fw) or (w == x):
                    v, fv = w, fw
                    w, fw = u, fu
                elif (fu <= fv) or (v == x) or (v == w):
                    v, fv = u, fu

            e = d if parabolic_accepted else e

        return steps


    def _animate_golden(self, interval_ms=1000):
        steps = self.gold_steps
        if not steps:
            return None

        x_vals = np.linspace(self.a, self.b, 400)
        y_vals = self.f(x_vals)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, y_vals, label='f(x)', color='blue')
        ax.set_title("Golden Section Search", fontsize=14)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, linestyle=':')
        ax.legend()

        a_line = ax.axvline(0, color='red', linestyle=':', lw=1.5, label='x1')
        b_line = ax.axvline(0, color='green', linestyle=':', lw=1.5, label='x2')

        a_point, = ax.plot([], [], 'o', color='gold')
        b_point, = ax.plot([], [], 'o', color='gold')
        c_point, = ax.plot([], [], 'o', color='gold', label='x3')
        d_point, = ax.plot([], [], 'o', color='orange', label='x4')
        conn, = ax.plot([], [], '--', color='purple', alpha=0.5)

        ax.legend()

        def update(i):
            a, b, c, d = steps[i]
            y_a, y_b, y_c, y_d = self.f(a), self.f(b), self.f(c), self.f(d)

            a_line.set_xdata([a])
            b_line.set_xdata([b])

            a_point.set_data([a], [y_a])
            b_point.set_data([b], [y_b])
            c_point.set_data([c], [y_c])
            d_point.set_data([d], [y_d])
            conn.set_data([a, c, d, b], [y_a, y_c, y_d, y_b])

            ax.set_title(f"Golden Section Search - Step {i+1}\n"
                         f"Interval: [{a:.4f}, {b:.4f}]")
            return a_line, b_line, a_point, b_point, c_point, d_point, conn

        anim = FuncAnimation(fig, update, frames=len(steps),
                             interval=interval_ms, blit=True, repeat=False)
        plt.close(fig)
        return HTML(anim.to_jshtml())


    def _animate_brent(self, interval_ms=1200):
        steps = self.brent_steps
        if not steps:
            return None

        x_vals = np.linspace(self.a, self.b, 800)
        y_vals = self.f(x_vals)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, y_vals, label='f(x)', color='blue')
        ax.set_title("Brent's Method (parabolic interpolation + golden fallback)", fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, linestyle=':')

        a_line = ax.axvline(0, color='red', linestyle=':', lw=1.5, label='a')
        b_line = ax.axvline(0, color='green', linestyle=':', lw=1.5, label='b')
        x_dot, = ax.plot([], [], 'ro', label='x (best)')
        w_dot, = ax.plot([], [], 'go', label='w')
        v_dot, = ax.plot([], [], 'bo', label='v')
        u_dot, = ax.plot([], [], marker='*', markersize=10, color='gold', label='u (candidate)')
        parabola_line, = ax.plot([], [], '--', color='purple', alpha=0.8, label='parabola (if stable)')

        annotation_texts = []
        ax.legend()

        def good_for_parabola(xs):
            xs = np.asarray(xs, dtype=float)
            min_spacing = np.min(np.abs(np.diff(np.sort(xs))))
            domain_width = abs(self.b - self.a) if self.b != self.a else 1.0
            return (min_spacing > 1e-10 * domain_width)

        def update(i):
            for t in annotation_texts:
                t.remove()
            annotation_texts.clear()

            a, b, x, w, v, u, method = steps[i]
            fa, fb, fx, fw, fv, fu = self.f(a), self.f(b), self.f(x), self.f(w), self.f(v), self.f(u)

            a_line.set_xdata([a])
            b_line.set_xdata([b])

            x_dot.set_data([x], [fx])
            w_dot.set_data([w], [fw])
            v_dot.set_data([v], [fv])
            u_dot.set_data([u], [fu])

            xs_to_fit = np.array([x, w, v])
            min_spacing = np.min(np.abs(np.diff(np.sort(xs_to_fit))))
            domain_width = abs(self.b - self.a) if self.b != self.a else 1.0

            xs_parabola = np.linspace(min(x, w, v) - 0.05*domain_width, max(x, w, v) + 0.05*domain_width, 300)
            draw_parabola = False
            if min_spacing > 1e-10 * domain_width:
                try:
                    coeffs = np.polyfit([x, w, v], [fx, fw, fv], 2)
                    parabola = np.polyval(coeffs, xs_parabola)
                    if np.all(np.isfinite(parabola)):
                        parabola_line.set_data(xs_parabola, parabola)
                        draw_parabola = True
                except Exception:
                    draw_parabola = False

            if not draw_parabola:
                parabola_line.set_data([], [])

            labels = [
                (a, fa, 'a'),
                (b, fb, 'b'),
                (x, fx, 'x'),
                (w, fw, 'w'),
                (v, fv, 'v'),
                (u, fu, f'u ({method[:3]})')
            ]

            ax.set_title(f"Brent's Method - Step {i+1}  (method: {method})\nInterval: [{a:.6f}, {b:.6f}]")
            return (a_line, b_line, x_dot, w_dot, v_dot, u_dot, parabola_line, *annotation_texts)

        anim = FuncAnimation(fig, update, frames=len(steps),
                             interval=interval_ms, blit=False, repeat=False)
        plt.close(fig)
        return HTML(anim.to_jshtml())

    def show_toggle_closed(self, interval_ms=1000):
        gold_anim_html = self._animate_golden(interval_ms)
        brent_anim_html = self._animate_brent(interval_ms)

        if gold_anim_html is None or brent_anim_html is None:
            print("Cannot create animations. Check initial interval [a, b].")
            return

        gold_content = widgets.HTML(value=gold_anim_html.data)
        brent_content = widgets.HTML(value=brent_anim_html.data)

        tab_container = widgets.Tab()
        tab_container.children = [gold_content, brent_content]
        tab_container.set_title(0, 'Golden Section Search')
        tab_container.set_title(1, "Brent's Method")

        display(tab_container)


class ScalarTracker:
    """Wraps a 1D objective function to record scalar evaluations during optimization."""
    def __init__(self, f):
        self.f = f
        self.x_history = []
        self.f_history = []

    def __call__(self, x):
        val = self.f(x)
        if np.ndim(x) == 0 or (isinstance(x, np.ndarray) and x.size == 1):
            x_scalar = float(np.squeeze(x))
            f_scalar = float(np.squeeze(val))
            self.x_history.append(x_scalar)
            self.f_history.append(f_scalar)
        return val


def plot_bracketed_progression(f, tracker, result=None, x_range=(-5, 5), x_true=None, title=None):
    """Plots a 2-panel Plotly figure showing function evaluations and convergence progression."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = f(x_vals)

    xs = np.array(tracker.x_history, dtype=float)
    ys = np.array(tracker.f_history, dtype=float)
    steps = np.arange(1, len(xs) + 1)

    if result is not None:
        x_min, y_min = float(result.x), float(result.fun)
    elif len(xs) > 0:
        idx = int(np.argmin(ys))
        x_min, y_min = xs[idx], ys[idx]
    else:
        x_min, y_min = 0.0, float(f(0.0))

    x_ref = float(x_true) if x_true is not None else x_min

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Objective Function & Evaluations", "Solver Progression (|xₖ - x*| & xₖ)"),
        horizontal_spacing=0.12,
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )

    # Left panel: f(x) curve
    fig.add_trace(
        go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)', line=dict(color='royalblue', width=2)),
        row=1, col=1
    )

    # Left panel: evaluated points colored by iteration
    if len(xs) > 0:
        hover_texts = [f"Eval {k}<br>x = {xk:.6g}<br>f(x) = {yk:.6g}" for k, xk, yk in zip(steps, xs, ys)]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode='markers+lines',
                name='Evaluations',
                line=dict(color='rgba(150, 150, 150, 0.35)', dash='dot', width=1),
                marker=dict(
                    size=8,
                    color=steps,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Eval #", len=0.85, x=-0.12),
                    line=dict(width=1, color='black')
                ),
                text=hover_texts,
                hoverinfo='text'
            ),
            row=1, col=1
        )

    # Left panel: Final minimum star
    fig.add_trace(
        go.Scatter(
            x=[x_min], y=[y_min],
            mode='markers',
            name=f'Minimum (x*={x_min:.4g})',
            marker=dict(color='crimson', size=13, symbol='star', line=dict(width=1, color='black'))
        ),
        row=1, col=1
    )

    # Right panel: x_k progression (primary y-axis) and |x_k - x*| (secondary log y-axis)
    if len(xs) > 0:
        fig.add_trace(
            go.Scatter(
                x=steps, y=xs,
                mode='lines+markers',
                name='Guess xₖ',
                line=dict(color='teal', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2, secondary_y=False
        )
        fig.add_hline(y=x_ref, line_dash='dash', line_color='crimson', opacity=0.6, row=1, col=2, secondary_y=False)

        raw_err = np.abs(xs - x_ref)
        if x_true is None and np.any(raw_err == 0):
            nonzero = raw_err[raw_err > 0]
            floor_val = 0.5 * np.min(nonzero) if len(nonzero) > 0 else 1e-12
            err = np.where(raw_err == 0, floor_val, raw_err)
        else:
            err = np.maximum(raw_err, 1e-16)

        fig.add_trace(
            go.Scatter(
                x=steps, y=err,
                mode='lines+markers',
                name='|xₖ - x*| (log)',
                line=dict(color='darkorange', width=2, dash='dot'),
                marker=dict(size=6, symbol='diamond')
            ),
            row=1, col=2, secondary_y=True
        )

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_xaxes(title_text="Evaluation Number (k)", row=1, col=2)
    fig.update_yaxes(title_text="Guess xₖ", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Error |xₖ - x*|", type="log", row=1, col=2, secondary_y=True)

    fig.update_layout(
        title=title or f"Solver Progression ({len(xs)} Function Evaluations)",
        width=950,
        height=430,
        margin=dict(l=90, r=60, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5)
    )
    fig.show()

