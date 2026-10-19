import numpy as np

def weierstrass(x, a=0.5, b=13, iterations=100):
    """
    Computes the Weierstrass function.
    
    Parameters:
    x (array-like): The points at which to evaluate the function.
    a (float): Parameter 0 < a < 1.
    b (float): Parameter such that ab > 1 + 3pi/2.
    iterations (int): Number of terms in the sum.
    
    Returns:
    y (ndarray): The function values.
    """
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    for n in range(iterations):
        y += (a**n) * np.cos((b**n) * np.pi * x)
    return y

import plotly.graph_objects as go
from scipy.optimize import root

def plot_and_find_roots(func):
    x = np.linspace(-10, 10, 400)
    y = func(x)

    # Find roots
    x0s = np.arange(-10, 10, 1)
    roots = []
    for x0 in x0s:
        r = root(func, x0=x0)
        if r.success:
            roots.append((r.x[0], r.fun[0]))

    # Remove duplicate roots (within tolerance)
    unique_roots = []
    tol = 1e-6
    for rx, ry in roots:
        if not any(abs(rx - ux) < tol for ux, _ in unique_roots):
            unique_roots.append((rx, ry))

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)'))
    fig.add_hline(y=0, line_dash='dash', line_color='black', annotation_text='x-axis')
    if unique_roots:
        rx_vals = [rx for rx, _ in unique_roots]
        fig.add_trace(go.Scatter(x=rx_vals, y=[0]*len(unique_roots), mode='markers', marker=dict(color='red', size=10), name='Roots'))
    fig.update_layout(title='Plot of f(x) and its Roots', xaxis_title='x', yaxis_title='f(x)', xaxis_range=[-10, 10], yaxis_range=[-10, 10])
    fig.show()

def complex_plot_with_shadow(func, title='Complex Plot of f(z)', roots=None, plot_range=(-3, 3)):
    real_range = np.linspace(plot_range[0], plot_range[1], 100)
    imag_range = np.linspace(plot_range[0], plot_range[1], 100)
    abs_val = np.empty((len(real_range), len(imag_range)))

    for i, real in enumerate(real_range):
        for j, imag in enumerate(imag_range):
            z = complex(real, imag)
            result = func(z)
            abs_val[i, j] = abs(result)

    surface = go.Surface(
        z=abs_val.T, 
        x=real_range, 
        y=imag_range, 
        colorscale='Viridis',
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    )
    
    data = [surface]
    
    if roots is not None:
        root_x = [r.real for r in roots]
        root_y = [r.imag for r in roots]
        # Calculate z for roots (which is abs(func(r)) = 0)
        root_z = [0 for r in roots]
        
        data.append(go.Scatter3d(
            x=root_x, y=root_y, z=root_z, 
            mode='markers', 
            marker=dict(color='red', size=5, symbol='circle'),
            name='Roots'
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title, 
        scene=dict(xaxis_title='Real', yaxis_title='Imaginary', zaxis_title='|f(z)|')
    )
    fig.show()


from scipy.linalg import companion

def plot_polynomial_roots_companion(n):
    """
    Creates a random polynomial of order n, solves for its roots 
    using the companion matrix method, and plots the polynomial 
    along with the roots as a complex surface with a shadow.
    """
    # Create random coefficients from -10 to 10
    # A polynomial of order n has n+1 coefficients. Let's make the leading coefficient non-zero.
    coeffs = np.random.uniform(-10, 10, n + 1)
    
    # Make sure leading coefficient is not too small
    if abs(coeffs[0]) < 1:
        coeffs[0] = 1.0 * np.sign(coeffs[0]) if coeffs[0] != 0 else 1.0
        
    poly_func = np.poly1d(coeffs)
    
    print("Polynomial:")
    print(poly_func)
    
    # Solve for roots using companion matrix
    comp_matrix = companion(coeffs)
    
    print("\nCompanion Matrix:")
    print(np.round(comp_matrix, 2))
    
    roots = np.linalg.eigvals(comp_matrix)
    
    print("\nRoots from eigenvalues of companion matrix:")
    print(np.round(roots, 2))
    
    # Determine plot range based on roots
    max_abs_root = max(abs(roots)) if len(roots) > 0 else 1.0
    plot_limit = max_abs_root * 1.5
    
    # --- Real-space plot ---
    x_real = np.linspace(-plot_limit, plot_limit, 400)
    y_real = poly_func(x_real)
    
    fig_real = go.Figure()
    fig_real.add_trace(go.Scatter(x=x_real, y=y_real, mode='lines', name='f(x)'))
    fig_real.add_hline(y=0, line_dash='dash', line_color='black', annotation_text='x-axis')
    
    # Find real roots (imaginary part near 0)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    if real_roots:
        fig_real.add_trace(go.Scatter(x=real_roots, y=[0]*len(real_roots), mode='markers', marker=dict(color='red', size=10), name='Real Roots'))
        
    fig_real.update_layout(title=f"Real Space Plot: Roots of Order {n} Polynomial", xaxis_title='x', yaxis_title='f(x)')
    fig_real.show()
    
    # Plot using complex_plot_with_shadow
    title = f"Complex Space Plot: Roots of Order {n} Polynomial"
    complex_plot_with_shadow(poly_func, title=title, roots=roots, plot_range=(-plot_limit, plot_limit))


def plot_secant(f, x0, x1, tolerance=1e-6, max_iterations=100, title='Secant Method'):
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np

    x_values = [x0, x1]
    for i in range(max_iterations):
        denom = f(x1) - f(x0)
        if denom == 0:
            break
        x_new = x1 - f(x1) * (x1 - x0) / denom
        x_values.append(x_new)
        if abs(f(x_new)) < tolerance:
            break
        x0 = x1
        x1 = x_new
    
    root = x_values[-1] if abs(f(x_values[-1])) < tolerance else None

    # Determine plot bounds
    min_x = min(x_values)
    max_x = max(x_values)
    padding = (max_x - min_x) * 0.2 if max_x > min_x else 0.5
    x = np.linspace(min_x - padding, max_x + padding, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(x), mode='lines', name='f(x)'))
    fig.add_trace(go.Scatter(x=[x_values[0], x_values[1]], y=[f(x_values[0]), f(x_values[1])], mode='markers', name='Initial guesses'))
    
    colors = px.colors.qualitative.Plotly
    
    for i in range(1, len(x_values) - 1):
        # Secant line intersecting x-axis
        fig.add_trace(go.Scatter(x=[x_values[i-1], x_values[i], x_values[i+1]], y=[f(x_values[i-1]), f(x_values[i]), 0], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
        # Vertical dotted line to the curve
        fig.add_trace(go.Scatter(x=[x_values[i+1], x_values[i+1]], y=[0, f(x_values[i+1])], mode='lines', line=dict(dash='dot', color='gray'), showlegend=False))
        
        color = colors[(i-1) % len(colors)]
        fig.add_trace(go.Scatter(x=[x_values[i+1]], y=[0], mode='markers', marker=dict(size=10, color=color), name=f'x_{i+1}'))
    
    if root:
        fig.add_trace(go.Scatter(x=[root], y=[0], mode='markers', marker=dict(color='green', size=10), name='Approximate root'))
    
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='f(x)')
    fig.show()
    return root, x_values

def plot_newton_raphson(f, df, x0, tolerance=1e-6, max_iterations=100, title='Newton-Raphson Method', xrange=None, yrange=None):
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    
    x_values = [x0]
    for i in range(max_iterations):
        d = df(x0)
        if d == 0:
            break
        x_new = x0 - f(x0) / d
        x_values.append(x_new)
        if abs(f(x_new)) < tolerance or abs(x_new - x0) < tolerance:
            break
        x0 = x_new

    root = x_values[-1] if abs(f(x_values[-1])) < tolerance or abs(x_values[-1] - x_values[-2]) < tolerance else None

    # Determine plot bounds
    if xrange is None:
        min_x = min(x_values)
        max_x = max(x_values)
        padding = (max_x - min_x) * 0.2 if max_x > min_x else 0.5
        x = np.linspace(min_x - padding, max_x + padding, 100)
    else:
        x = np.linspace(xrange[0], xrange[1], 100)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(x), mode='lines', name='f(x)'))
    fig.add_trace(go.Scatter(x=[x_values[0]], y=[f(x_values[0])], mode='markers', name='Initial guess'))
    
    colors = px.colors.qualitative.Plotly
    
    for i in range(len(x_values) - 1):
        # Tangent line intersecting x-axis
        fig.add_trace(go.Scatter(x=[x_values[i], x_values[i+1]], y=[f(x_values[i]), 0], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
        # Vertical dotted line to the curve
        fig.add_trace(go.Scatter(x=[x_values[i+1], x_values[i+1]], y=[0, f(x_values[i+1])], mode='lines', line=dict(dash='dot', color='gray'), showlegend=False))
        
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=[x_values[i+1]], y=[0], mode='markers', marker=dict(size=10, color=color), name=f'x_{i+1}'))
    
    if root:
        fig.add_trace(go.Scatter(x=[root], y=[0], mode='markers', marker=dict(color='green', size=10), name='Approximate root'))
    
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='f(x)')
    if xrange:
        fig.update_layout(xaxis_range=xrange)
    if yrange:
        fig.update_layout(yaxis_range=yrange)
        
    fig.show()
    return root, x_values

def plot_basins_of_attraction(f, df, known_roots, title, real_range=(-2, 2), imag_range=(-2, 2), grid_size=500, th=1e-3):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.optimize import newton

    real_values = np.linspace(real_range[0], real_range[1], grid_size)
    imag_values = np.linspace(imag_range[0], imag_range[1], grid_size)
    z_grid = np.array([[complex(r, i) for r in real_values] for i in imag_values])

    roots = newton(f, z_grid, fprime=df)

    colors = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            if roots[i, j] is None:
                colors[i, j] = 0
            else:
                for idx, r in enumerate(known_roots):
                    if abs(roots[i, j] - r) < th:
                        colors[i, j] = idx + 1
                        break
                        
    fig = go.Figure(data=go.Heatmap(z=colors, x=real_values, y=imag_values, colorscale='Viridis', showscale=False))
    fig.update_layout(title=title, xaxis_title='Real', yaxis_title='Imaginary', width=600, height=600)
    fig.show()

def newton_raphson(f, df, x0, max_iter=8, tolerance=1e-6):
  x = x0
  guesses = [x]
  for i in range(max_iter):
    d = df(x)
    if d == 0:
        break
    x_new = x - f(x) / d
    guesses.append(x_new)
    if abs(x_new - x) < tolerance:
      return x_new, guesses
    x = x_new
  return None, guesses
