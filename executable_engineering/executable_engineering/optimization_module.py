import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
from scipy.linalg import solve


def visual_golden_section(f, x1, x3, tol=1e-5, max_iter=None):

    phi = (1 + np.sqrt(5)) / 2
    invphi = 1 / phi

    states = []
    all_evals = []

    initial_width = abs(x3 - x1)

    if initial_width <= tol:
        required_iter = 0
    else:
        required_iter = int(
            np.ceil(np.log(tol / initial_width) / np.log(invphi))
        )

    if max_iter is not None:
        required_iter = min(required_iter, max_iter)

    curr_x1 = x1
    curr_x3 = x3

    curr_x2 = curr_x3 - invphi * (curr_x3 - curr_x1)
    curr_x4 = curr_x1 + invphi * (curr_x3 - curr_x1)

    f1 = f(curr_x1)
    f3 = f(curr_x3)
    f2 = f(curr_x2)
    f4 = f(curr_x4)

    all_evals.extend([
        curr_x1,
        curr_x2,
        curr_x4,
        curr_x3
    ])

    states.append({
        'iter': 0,
        'active': [
            (curr_x1, f1, 'x1'),
            (curr_x2, f2, 'x2'),
            (curr_x4, f4, 'x4'),
            (curr_x3, f3, 'x3')
        ],
        'history': list(dict.fromkeys(all_evals))
    })

    for i in range(1, required_iter + 1):

        if abs(curr_x3 - curr_x1) <= tol:
            break

        if f4 < f2:

            curr_x1, f1 = curr_x2, f2
            curr_x2, f2 = curr_x4, f4

            curr_x4 = curr_x1 + invphi * (curr_x3 - curr_x1)
            f4 = f(curr_x4)

            all_evals.append(curr_x4)

        else:

            curr_x3, f3 = curr_x4, f4
            curr_x4, f4 = curr_x2, f2

            curr_x2 = curr_x3 - invphi * (curr_x3 - curr_x1)
            f2 = f(curr_x2)

            all_evals.append(curr_x2)

        states.append({
            'iter': i,
            'active': [
                (curr_x1, f1, 'x1'),
                (curr_x2, f2, 'x2'),
                (curr_x4, f4, 'x4'),
                (curr_x3, f3, 'x3')
            ],
            'history': list(dict.fromkeys(all_evals))
        })

    x_min = min(all_evals)
    x_max = max(all_evals)

    margin = (x_max - x_min) * 0.15

    if margin == 0:
        margin = 1

    x_vals = np.linspace(
        x_min - margin,
        x_max + margin,
        400
    )

    y_vals = f(x_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='f(x)',
        line=dict(
            color='lightgray',
            width=2
        ),
        hoverinfo='skip'
    ))

    init_state = states[0]

    ax = [pt[0] for pt in init_state['active']]
    ay = [pt[1] for pt in init_state['active']]
    alabels = [pt[2] for pt in init_state['active']]

    hx = [
        x for x in init_state['history']
        if x not in ax
    ]

    hy = [f(x) for x in hx]

    fig.add_trace(go.Scatter(
        x=hx,
        y=hy,
        mode='markers',
        name='Past Evaluations',
        marker=dict(
            color='gray',
            size=7,
            opacity=0.5
        )
    ))

    fig.add_trace(go.Scatter(
        x=ax,
        y=ay,
        mode='markers+text',
        name='Active Bracket',
        text=alabels,
        textposition='top center',
        textfont=dict(
            color='black',
            size=13
        ),
        marker=dict(
            color='crimson',
            size=12,
            symbol='circle',
            line=dict(
                color='black',
                width=1
            )
        )
    ))

    frames = []

    for state in states:

        cur_ax = [pt[0] for pt in state['active']]
        cur_ay = [pt[1] for pt in state['active']]
        cur_alabels = [pt[2] for pt in state['active']]

        cur_hx = [
            x for x in state['history']
            if x not in cur_ax
        ]

        cur_hy = [f(x) for x in cur_hx]

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=cur_hx,
                        y=cur_hy
                    ),
                    go.Scatter(
                        x=cur_ax,
                        y=cur_ay,
                        text=cur_alabels
                    )
                ],
                name=f"Iter {state['iter']}",
                traces=[1, 2]
            )
        )

    fig.frames = frames

    sliders = [{
        "active": 0,
        "y": -0.05,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {
                "size": 16
            },
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {
            "duration": 0
        },
        "pad": {
            "b": 10,
            "t": 20
        },
        "steps": [
            {
                "args": [
                    [frame.name],
                    {
                        "frame": {
                            "duration": 0,
                            "redraw": True
                        },
                        "mode": "immediate"
                    }
                ],
                "label": str(k),
                "method": "animate"
            }
            for k, frame in enumerate(frames)
        ]
    }]

    updatemenus = [{
        "buttons": [
            {
                "args": [
                    None,
                    {
                        "frame": {
                            "duration": 600,
                            "redraw": True
                        },
                        "fromcurrent": True
                    }
                ],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [
                    [None],
                    {
                        "frame": {
                            "duration": 0,
                            "redraw": True
                        },
                        "mode": "immediate"
                    }
                ],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {
            "r": 10,
            "t": 10
        },
        "showactive": False,
        "type": "buttons",
        "x": 0.5,
        "xanchor": "center",
        "y": -0.32,
        "yanchor": "top"
    }]

    fig.update_layout(
        title=(
            f"Golden Section Search"
        ),
        xaxis_title="x",
        yaxis_title="f(x)",
        width=700,
        height=540,
        margin=dict(
            l=25,
            r=25,
            t=50,
            b=160
        ),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea'
        ),
        updatemenus=updatemenus,
        sliders=sliders
    )

    return fig
  

def visual_powell_2d(
    f,
    point0,
    h1=(1, 0),
    h2=(0, 1),
    tol=1e-6,
    max_iter=100
):


    point = np.array(point0, dtype=float)
    h1 = np.array(h1, dtype=float)
    h2 = np.array(h2, dtype=float)

    def evaluate(pt):
        return f(pt[0], pt[1])

    def line_search(pt, direction):
        if np.linalg.norm(direction) < 1e-12:
            return pt.copy()

        res = minimize_scalar(
            lambda alpha: evaluate(pt + alpha * direction),
            method="brent"
        )

        return pt + res.x * direction

    
    points = [point.copy()]

    directions = [h1, h2]

    converged = False


    for iteration in range(max_iter):

        iteration_start = point.copy()
        f_start = evaluate(point)

        for direction in directions:
            point = line_search(point, direction)
            points.append(point.copy())

        new_direction = point - iteration_start

        if np.linalg.norm(new_direction) < tol:
            converged = True
            break

        point_before_new_direction = point.copy()

        point = line_search(point, new_direction)
        points.append(point.copy())

        f_new = evaluate(point)

        point_change = np.linalg.norm(
            point - point_before_new_direction
        )

        objective_change = abs(f_new - f_start)

        if point_change < tol and objective_change < tol:
            converged = True
            break

        directions = [
            directions[1],
            new_direction
        ]

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    labels = [str(i) for i in range(len(points))]

    labels[-1] = "Min"

    padding = 2.0

    x_min = min(x_coords) - padding
    x_max = max(x_coords) + padding

    y_min = min(y_coords) - padding
    y_max = max(y_coords) + padding

    x_values = np.linspace(x_min, x_max, 200)
    y_values = np.linspace(y_min, y_max, 200)

    X, Y = np.meshgrid(x_values, y_values)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(X[i, j], Y[i, j])

    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=Z,
            colorscale="Blues_r",
            showscale=False,
            contours=dict(
                start=np.min(Z),
                end=np.max(Z),
                size=(np.max(Z) - np.min(Z)) / 15
            ),
            line=dict(width=1),
            name="Objective Contour"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines",
            name="Search Path",
            line=dict(
                color="black",
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            name="Evaluation Points",
            text=labels,
            textposition="top center",
            textfont=dict(
                size=12,
                color="black"
            ),
            marker=dict(
                color=["black"] * (len(points) - 1) + ["red"],
                size=[8] * (len(points) - 1) + [14],
                symbol=["circle"] * (len(points) - 1) + ["star"]
            ),
            hoverinfo="text",
            hovertext=[
                (
                    f"Point {label}"
                    f"<br>x: {x:.6f}"
                    f"<br>y: {y:.6f}"
                    f"<br>f(x,y): {f(x, y):.6f}"
                )
                for label, x, y in zip(
                    labels,
                    x_coords,
                    y_coords
                )
            ]
        )
    )

    title = (f"Powell's Method")
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        width=700,
        height=525,
        margin=dict(
            l=25,
            r=25,
            t=50,
            b=25
        ),
        plot_bgcolor="white"
    )

    return fig
  

def newton_method(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=100):
  x = x0
  print(x)
  guesses = [x]
  for _ in range(max_iter):
    grad = grad_f(x)
    hess = hessian_f(x)

    # ~~ What goes here?

    ###
    delta_x = solve(hess, -grad, assume_a = 'sym')
    ###
    x = x + delta_x
    guesses.append(x)
    print(x)
    if np.linalg.norm(grad) < tol:
      break

  # Create a surface plot of the function
  x = np.linspace(-2, 2, 100)
  y = np.linspace(-2, 2, 100)
  X, Y = np.meshgrid(x, y)
  Z = f([X, Y])

  fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])

  # Add markers for each guess
  for guess in guesses:
    fig.add_trace(go.Scatter3d(
        x=[guess[0]],
        y=[guess[1]],
        z=[f(guess)],
        mode='markers',
        marker=dict(
            size=5,
            color='red'
        )
    ))

  fig.update_layout(
      title='Newton\'s Method Optimization',
      height = 600,
      scene=dict(
          xaxis_title='x',
          yaxis_title='y',
          zaxis_title='f(x,y)'
      )
  )
  fig.show()


