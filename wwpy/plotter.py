# plotter.py
# Functions and classes for visualization


# Template for a possible plotter idea.


import plotly.graph_objects as go
import numpy as np

def create_mesh_grid(x_range, y_range, z_range, step=1):
    """
    Creates grid lines along each axis within the specified ranges.

    Parameters:
        x_range (tuple): (min, max) range for x-axis
        y_range (tuple): (min, max) range for y-axis
        z_range (tuple): (min, max) range for z-axis
        step (int): spacing between grid lines

    Returns:
        x_mesh, y_mesh, z_mesh: Lists of Scatter3d traces for each axis
    """
    x_mesh = []
    y_mesh = []
    z_mesh = []

    # X Mesh: Lines parallel to X-axis (constant Y and Z)
    for y in np.arange(y_range[0], y_range[1] + step, step):
        for z in np.arange(z_range[0], z_range[1] + step, step):
            x_line = np.linspace(x_range[0], x_range[1], 2)
            y_line = [y, y]
            z_line = [z, z]
            x_mesh.append(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(color='red', width=2),
                name='X Mesh',
                legendgroup='X Mesh',
                showlegend=False  # Hide individual lines from legend
            ))

    # Y Mesh: Lines parallel to Y-axis (constant X and Z)
    for x in np.arange(x_range[0], x_range[1] + step, step):
        for z in np.arange(z_range[0], z_range[1] + step, step):
            y_line = np.linspace(y_range[0], y_range[1], 2)
            x_line = [x, x]
            z_line = [z, z]
            y_mesh.append(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(color='green', width=2),
                name='Y Mesh',
                legendgroup='Y Mesh',
                showlegend=False  # Hide individual lines from legend
            ))

    # Z Mesh: Lines parallel to Z-axis (constant X and Y)
    for x in np.arange(x_range[0], x_range[1] + step, step):
        for y in np.arange(y_range[0], y_range[1] + step, step):
            z_line = np.linspace(z_range[0], z_range[1], 2)
            x_line = [x, x]
            y_line = [y, y]
            z_mesh.append(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Z Mesh',
                legendgroup='Z Mesh',
                showlegend=False  # Hide individual lines from legend
            ))

    return x_mesh, y_mesh, z_mesh

def add_legend_traces(fig):
    """
    Adds dummy traces to the figure for legend entries.

    Parameters:
        fig (go.Figure): The Plotly figure object to which legend traces are added.
    """
    # X Mesh Legend Trace
    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='lines',
        line=dict(color='red', width=2),
        name='X Mesh',
        legendgroup='X Mesh',
        showlegend=True
    ))

    # Y Mesh Legend Trace
    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='lines',
        line=dict(color='green', width=2),
        name='Y Mesh',
        legendgroup='Y Mesh',
        showlegend=True
    ))

    # Z Mesh Legend Trace
    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Z Mesh',
        legendgroup='Z Mesh',
        showlegend=True
    ))

def generate_mesh_grid_html(output_file='mesh_grid.html'):
    """
    Generates an interactive 3D mesh grid plot and saves it as an HTML file.

    Parameters:
        output_file (str): The filename for the output HTML file.
    """
    # Define the range for each axis
    x_range = (0, 10)
    y_range = (0, 10)
    z_range = (0, 10)
    step = 2  # spacing between grid lines

    # Create mesh grids
    x_mesh, y_mesh, z_mesh = create_mesh_grid(x_range, y_range, z_range, step)

    # Create the figure
    fig = go.Figure()

    # Add X Mesh lines to the figure
    for trace in x_mesh:
        fig.add_trace(trace)

    # Add Y Mesh lines to the figure
    for trace in y_mesh:
        fig.add_trace(trace)

    # Add Z Mesh lines to the figure
    for trace in z_mesh:
        fig.add_trace(trace)

    # Add legend traces
    add_legend_traces(fig)

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range, title='X Axis'),
            yaxis=dict(range=y_range, title='Y Axis'),
            zaxis=dict(range=z_range, title='Z Axis'),
            aspectmode='cube'
        ),
        title='3D Mesh Grid with Interactive Legend',
        legend=dict(
            itemsizing='constant'
        )
    )

    # Save the figure to an HTML file
    fig.write_html('examples/'+output_file)
    print(f"Interactive 3D mesh grid has been saved to {output_file}")

if __name__ == "__main__":
    generate_mesh_grid_html()
