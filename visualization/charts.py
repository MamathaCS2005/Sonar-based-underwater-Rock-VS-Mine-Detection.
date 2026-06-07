"""
Chart visualizations for the underwater mine detection dashboard.
Includes pie charts, feature importance, and confusion matrix plots.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_class_distribution_pie(df):
    """
    Create pie chart showing Mine vs Rock distribution.
    
    Args:
        df: pandas DataFrame with 'label' column
    
    Returns:
        plotly figure
    """
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Class"] = counts["Class"].map({"M": "Mine", "R": "Rock"})

    fig = px.pie(
        counts,
        values="Count",
        names="Class",
        color="Class",
        color_discrete_map={"Mine": "#FF4444", "Rock": "#44AAFF"},
        title="Object Class Distribution"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font_size=16
    )
    return fig


def create_feature_importance_chart(importance_dict, top_n=15):
    """
    Create horizontal bar chart of feature importance.
    
    Args:
        importance_dict: dict of feature_name -> importance_score
        top_n: number of top features to display
    
    Returns:
        plotly figure
    """
    # Get top N features
    items = list(importance_dict.items())[:top_n]
    features = [item[0] for item in items]
    scores = [item[1] for item in items]

    fig = go.Figure(go.Bar(
        x=scores[::-1],
        y=features[::-1],
        orientation="h",
        marker=dict(
            color=scores[::-1],
            colorscale="Viridis"
        )
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importance (Random Forest)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font_size=16,
        height=500
    )
    return fig


def create_confusion_matrix_chart(cm):
    """
    Create confusion matrix heatmap.
    
    Args:
        cm: 2x2 confusion matrix (list of lists or numpy array)
    
    Returns:
        plotly figure
    """
    cm = np.array(cm)
    labels = ["Rock", "Mine"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font_size=16,
        height=400,
        width=400
    )
    return fig


def create_confidence_gauge(confidence, prediction):
    """
    Create a gauge chart showing prediction confidence.
    
    Args:
        confidence: float, confidence percentage
        prediction: str, "Mine" or "Rock"
    
    Returns:
        plotly figure
    """
    color = "#FF4444" if prediction == "Mine" else "#44AAFF"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        title={"text": f"Detection Confidence ({prediction})", "font": {"color": "white"}},
        number={"suffix": "%", "font": {"color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "white",
            "steps": [
                {"range": [0, 60], "color": "#1a3a1a"},
                {"range": [60, 85], "color": "#3a3a1a"},
                {"range": [85, 100], "color": "#3a1a1a"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": confidence
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=300
    )
    return fig
