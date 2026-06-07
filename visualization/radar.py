"""
Radar visualization for underwater mine detection.
Creates a radar-style plot showing ship position, mine location, and safe direction.
"""
import plotly.graph_objects as go
import numpy as np


def create_radar_plot(mine_distance, mine_angle, safe_angle):
    """
    Create radar-style visualization showing:
    - Ship at center (0,0)
    - Mine position based on distance and angle
    - Safe navigation direction
    
    Args:
        mine_distance: float, distance to mine in meters
        mine_angle: float, angle to mine in degrees (0-180)
        safe_angle: float, recommended safe direction in degrees
    
    Returns:
        plotly figure
    """
    # Convert mine position to cartesian
    mine_rad = np.radians(mine_angle)
    mine_x = mine_distance * np.cos(mine_rad)
    mine_y = mine_distance * np.sin(mine_rad)

    # Safe direction indicator (at edge of radar)
    safe_rad = np.radians(safe_angle)
    safe_x = 110 * np.cos(safe_rad)
    safe_y = 110 * np.sin(safe_rad)

    # Create radar rings
    theta = np.linspace(0, 2 * np.pi, 100)

    fig = go.Figure()

    # Draw range rings
    for r in [25, 50, 75, 100]:
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta),
            y=r * np.sin(theta),
            mode="lines",
            line=dict(color="rgba(0, 255, 0, 0.2)", width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Draw crosshairs
    fig.add_trace(go.Scatter(
        x=[-110, 110], y=[0, 0],
        mode="lines",
        line=dict(color="rgba(0, 255, 0, 0.15)", width=1),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-110, 110],
        mode="lines",
        line=dict(color="rgba(0, 255, 0, 0.15)", width=1),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Ship position (center)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=15, color="#00FF00", symbol="triangle-up"),
        text=["SHIP"],
        textposition="bottom center",
        textfont=dict(color="#00FF00", size=10),
        name="Ship Position",
        hovertemplate="Ship<br>Position: Center<extra></extra>"
    ))

    # Mine position
    fig.add_trace(go.Scatter(
        x=[mine_x], y=[mine_y],
        mode="markers+text",
        marker=dict(size=18, color="#FF0000", symbol="x"),
        text=["MINE"],
        textposition="top center",
        textfont=dict(color="#FF0000", size=10),
        name=f"Mine ({mine_distance}m, {mine_angle}°)",
        hovertemplate=f"Mine Detected<br>Distance: {mine_distance}m<br>Angle: {mine_angle}°<extra></extra>"
    ))

    # Line from ship to mine (threat vector)
    fig.add_trace(go.Scatter(
        x=[0, mine_x], y=[0, mine_y],
        mode="lines",
        line=dict(color="#FF0000", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Safe direction arrow
    fig.add_trace(go.Scatter(
        x=[0, safe_x], y=[0, safe_y],
        mode="lines+markers",
        line=dict(color="#00FF00", width=3),
        marker=dict(size=[0, 12], symbol=["circle", "arrow-up"],
                    color="#00FF00", angle=safe_angle - 90),
        name=f"Safe Direction ({safe_angle}°)",
        hovertemplate=f"Safe Direction<br>Angle: {safe_angle}°<extra></extra>"
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text="SONAR RADAR DISPLAY",
            font=dict(color="#00FF00", size=16)
        ),
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(color="#00FF00"),
        xaxis=dict(
            range=[-130, 130],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y"
        ),
        yaxis=dict(
            range=[-130, 130],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#00FF00",
            borderwidth=1,
            font=dict(color="#00FF00")
        )
    )

    # Add range labels
    for r in [25, 50, 75, 100]:
        fig.add_annotation(
            x=r, y=-5,
            text=f"{r}m",
            showarrow=False,
            font=dict(color="rgba(0,255,0,0.5)", size=9)
        )

    return fig


def create_mini_radar(mine_distance, mine_angle):
    """
    Create a simplified mini radar for compact display.
    
    Args:
        mine_distance: float, distance to mine in meters
        mine_angle: float, angle in degrees
    
    Returns:
        plotly figure
    """
    mine_rad = np.radians(mine_angle)
    mine_x = mine_distance * np.cos(mine_rad)
    mine_y = mine_distance * np.sin(mine_rad)

    theta = np.linspace(0, 2 * np.pi, 100)

    fig = go.Figure()

    # Single ring
    fig.add_trace(go.Scatter(
        x=100 * np.cos(theta),
        y=100 * np.sin(theta),
        mode="lines",
        line=dict(color="rgba(0,255,0,0.3)", width=1),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Ship
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=10, color="#00FF00", symbol="triangle-up"),
        showlegend=False
    ))

    # Mine
    fig.add_trace(go.Scatter(
        x=[mine_x], y=[mine_y],
        mode="markers",
        marker=dict(size=14, color="#FF0000", symbol="x"),
        showlegend=False
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-120, 120], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
        yaxis=dict(range=[-120, 120], showgrid=False, zeroline=False, showticklabels=False),
        height=250,
        width=250,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    return fig
