from __future__ import annotations

import plotly.graph_objects as go


def plot_surface_plotly(GX, GY, GZ, x, y, z, ids, pops, title="pygenoscape"):
    surf = go.Surface(x=GX, y=GY, z=GZ, opacity=0.95, showscale=True)
    pts = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        text=[f"{i} ({p})" for i, p in zip(ids, pops)],
        marker=dict(size=4),
        name="samples",
    )
    fig = go.Figure(data=[surf, pts])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Height",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig