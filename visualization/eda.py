"""
Exploratory Data Analysis visualizations for the sonar dataset.
Generates correlation heatmaps, distribution plots, and statistical summaries.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_correlation_heatmap(df):
    """
    Create correlation matrix heatmap for sonar features.
    
    Args:
        df: pandas DataFrame with feature columns
    
    Returns:
        plotly figure
    """
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    # Use subset for readability (every 5th feature)
    subset_cols = feature_cols[::5]
    corr_matrix = df[subset_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=subset_cols,
        y=subset_cols,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        hovertemplate="Feature X: %{x}<br>Feature Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap (Sampled)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font_size=16,
        height=500
    )
    return fig


def create_feature_distribution(df, n_features=8):
    """
    Create distribution plots for selected features.
    
    Args:
        df: pandas DataFrame
        n_features: number of features to plot
    
    Returns:
        plotly figure
    """
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    selected = feature_cols[:n_features]

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=selected
    )

    for i, col in enumerate(selected):
        row = i // 4 + 1
        col_idx = i % 4 + 1

        # Mine distribution
        mine_data = df[df["label"] == "M"][col]
        rock_data = df[df["label"] == "R"][col]

        fig.add_trace(
            go.Histogram(
                x=mine_data, name="Mine", opacity=0.7,
                marker_color="#FF4444", showlegend=(i == 0),
                nbinsx=20
            ),
            row=row, col=col_idx
        )
        fig.add_trace(
            go.Histogram(
                x=rock_data, name="Rock", opacity=0.7,
                marker_color="#44AAFF", showlegend=(i == 0),
                nbinsx=20
            ),
            row=row, col=col_idx
        )

    fig.update_layout(
        title="Feature Distributions by Class",
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font_size=16,
        height=500,
        showlegend=True
    )
    return fig


def create_class_count_plot(df):
    """
    Create count plot for class distribution.
    
    Args:
        df: pandas DataFrame with 'label' column
    
    Returns:
        plotly figure
    """
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Class"] = counts["Class"].map({"M": "Mine", "R": "Rock"})

    fig = px.bar(
        counts,
        x="Class",
        y="Count",
        color="Class",
        color_discrete_map={"Mine": "#FF4444", "Rock": "#44AAFF"},
        title="Class Distribution (Count)"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font_size=16,
        showlegend=False
    )
    return fig


def create_signal_strength_plot(df):
    """
    Create average signal strength across all 60 features by class.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        plotly figure
    """
    feature_cols = [col for col in df.columns if col.startswith("feature_")]

    mine_mean = df[df["label"] == "M"][feature_cols].mean()
    rock_mean = df[df["label"] == "R"][feature_cols].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 61)),
        y=mine_mean.values,
        mode="lines",
        name="Mine",
        line=dict(color="#FF4444", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, 61)),
        y=rock_mean.values,
        mode="lines",
        name="Rock",
        line=dict(color="#44AAFF", width=2)
    ))
    fig.update_layout(
        title="Average Signal Strength by Feature (Mine vs Rock)",
        xaxis_title="Feature Index",
        yaxis_title="Mean Signal Strength",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font_size=16,
        height=400
    )
    return fig


def create_dataset_summary(df):
    """
    Generate dataset summary statistics.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        dict with summary info
    """
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    return {
        "total_samples": len(df),
        "n_features": len(feature_cols),
        "n_mines": int((df["label"] == "M").sum()),
        "n_rocks": int((df["label"] == "R").sum()),
        "feature_mean_range": f"{df[feature_cols].mean().min():.4f} - {df[feature_cols].mean().max():.4f}",
        "feature_std_range": f"{df[feature_cols].std().min():.4f} - {df[feature_cols].std().max():.4f}"
    }
