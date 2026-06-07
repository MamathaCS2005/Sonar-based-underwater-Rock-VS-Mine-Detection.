"""
AI-Based Underwater Mine Detection and Analysis System
Main Streamlit Dashboard Application

A defense-style intelligent decision support system that analyzes sonar signals
to classify underwater objects as Mine or Rock using Random Forest ML model.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Import project modules
from model.predict import MineDetector
from utils.risk_analysis import calculate_risk_level
from utils.direction import estimate_direction, estimate_distance, calculate_safe_direction
from utils.alerts import get_alert_config, generate_alert_html
from visualization.charts import (
    create_class_distribution_pie,
    create_feature_importance_chart,
    create_confusion_matrix_chart,
    create_confidence_gauge
)
from visualization.radar import create_radar_plot
from visualization.eda import (
    create_correlation_heatmap,
    create_feature_distribution,
    create_class_count_plot,
    create_signal_strength_plot,
    create_dataset_summary
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Underwater Mine Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - DARK DEFENSE THEME
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0e14;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1a3a2a;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #1a3a2a;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Headers */
    h1 {
        color: #00FF88 !important;
    }
    h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Status indicator */
    .status-online {
        color: #00FF00;
        font-weight: bold;
    }
    
    /* Alert animations */
    @keyframes flash {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .flash-alert {
        animation: flash 1s infinite;
    }
    
    /* Detection history table */
    .dataframe {
        background-color: #0d1117 !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #00FF88;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "detector" not in st.session_state:
    st.session_state.detector = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_data
def load_dataset():
    """Load the sonar dataset."""
    data_path = os.path.join("data", "sonar.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


def load_model():
    """Load the trained model."""
    try:
        detector = MineDetector()
        st.session_state.detector = detector
        st.session_state.model_loaded = True
        return detector
    except FileNotFoundError:
        st.session_state.model_loaded = False
        return None


def run_detection(features, detector):
    """Run full detection pipeline on given features."""
    # Predict
    result = detector.predict(features)

    # Direction & Distance
    direction = estimate_direction(features)
    distance = estimate_distance(features)

    # Risk analysis
    risk = calculate_risk_level(result["confidence"], result["prediction"])

    # Safe navigation
    safe_nav = calculate_safe_direction(direction)

    # Alert config
    alert = get_alert_config(risk["level"])

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    detection = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probability_mine": result["probability_mine"],
        "probability_rock": result["probability_rock"],
        "direction": direction,
        "distance": distance,
        "risk_level": risk["level"],
        "risk_color": risk["color"],
        "risk_description": risk["description"],
        "risk_action": risk["action"],
        "safe_angle": safe_nav["safe_angle"],
        "safe_cardinal": safe_nav["cardinal_direction"],
        "safe_recommendation": safe_nav["recommendation"],
        "alert": alert,
        "timestamp": timestamp
    }

    # Add to history
    st.session_state.detection_history.insert(0, detection)

    return detection


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # ----------------------------------------------------------
    # LOAD MODEL & DATA (before rendering status)
    # ----------------------------------------------------------
    df = load_dataset()
    detector = load_model()

    # ----------------------------------------------------------
    # TOP SECTION: Header & System Status
    # ----------------------------------------------------------
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown("# 🎯 DeepMine Sentinel")
        st.markdown("*Intelligent Sonar Analysis & Threat Assessment Platform*")
    with col_status:
        st.markdown("### System Status")
        if st.session_state.model_loaded:
            st.markdown('<p class="status-online">● ONLINE</p>', unsafe_allow_html=True)
        else:
            st.markdown("🔴 MODEL NOT LOADED")

    st.markdown("---")

    if df is None:
        st.error("⚠️ Dataset not found! Run `python data/download_dataset.py` first.")
        st.stop()

    if detector is None:
        st.error("⚠️ Model not found! Run `python model/train_model.py` first.")
        st.info("Steps to initialize:\n1. `python data/download_dataset.py`\n2. `python model/train_model.py`")
        st.stop()

    # ----------------------------------------------------------
    # SIDEBAR
    # ----------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🛠️ Control Panel")
        st.markdown("---")

        # Dataset info
        st.markdown("### 📊 Dataset Info")
        summary = create_dataset_summary(df)
        st.metric("Total Samples", summary["total_samples"])
        st.metric("Mines", summary["n_mines"])
        st.metric("Rocks", summary["n_rocks"])
        st.metric("Features", summary["n_features"])

        st.markdown("---")
        st.markdown("### 🔍 Detection Mode")
        detection_mode = st.radio(
            "Select input method:",
            ["Random Sample", "Upload CSV", "Manual Input"]
        )

        st.markdown("---")
        st.markdown("### 📈 Model Metrics")
        metadata_path = os.path.join("model", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            metrics = metadata.get("metrics", {})
            st.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
            st.metric("Precision", f"{metrics.get('precision', 0) * 100:.1f}%")
            st.metric("Recall", f"{metrics.get('recall', 0) * 100:.1f}%")
            st.metric("F1 Score", f"{metrics.get('f1_score', 0) * 100:.1f}%")

    # ----------------------------------------------------------
    # DETECTION INPUT
    # ----------------------------------------------------------
    detection = None
    features = None

    if detection_mode == "Random Sample":
        if st.button("🎯 Run Random Detection", type="primary", use_container_width=True):
            feature_cols = [col for col in df.columns if col.startswith("feature_")]
            sample = df.sample(1)
            features = sample[feature_cols].values[0]
            detection = run_detection(features, detector)

    elif detection_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload sonar signal CSV", type=["csv"])
        if uploaded_file is not None:
            upload_df = pd.read_csv(uploaded_file)
            feature_cols = [col for col in upload_df.columns if col.startswith("feature_")]
            if len(feature_cols) == 60:
                features = upload_df[feature_cols].iloc[0].values
                detection = run_detection(features, detector)
            else:
                st.error("CSV must contain 60 feature columns (feature_1 to feature_60)")

    elif detection_mode == "Manual Input":
        st.markdown("#### Enter 60 sonar feature values (comma-separated):")
        manual_input = st.text_area("Features:", height=100,
                                     placeholder="0.02, 0.03, 0.04, ... (60 values)")
        if st.button("🔍 Analyze Signal") and manual_input:
            try:
                values = [float(x.strip()) for x in manual_input.split(",")]
                if len(values) == 60:
                    features = np.array(values)
                    detection = run_detection(features, detector)
                else:
                    st.error(f"Expected 60 values, got {len(values)}")
            except ValueError:
                st.error("Invalid input. Enter numeric values separated by commas.")

    # ----------------------------------------------------------
    # DISPLAY DETECTION RESULTS
    # ----------------------------------------------------------
    if detection:
        # ALERT SECTION
        alert_html = generate_alert_html(detection["alert"])
        st.markdown(alert_html, unsafe_allow_html=True)

        # Play audio alert for HIGH risk
        if detection["risk_level"] == "HIGH":
            buzzer_path = os.path.join("assets", "buzzer.wav")
            if os.path.exists(buzzer_path):
                st.audio(buzzer_path)

        st.markdown("")

        # SECTION 1 & 2: Detection Result + Direction/Distance
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🎯 Detection Result")
            pred_emoji = "💣" if detection["prediction"] == "Mine" else "🪨"
            st.metric(
                "Object Classified",
                f"{pred_emoji} {detection['prediction']}",
            )
            st.metric("Confidence", f"{detection['confidence']}%")
            st.caption(f"Timestamp: {detection['timestamp']}")

        with col2:
            st.markdown("### 🧭 Direction & Distance")
            st.metric("Direction", f"{detection['direction']}°")
            st.metric("Distance", f"{detection['distance']} m")

        with col3:
            st.markdown("### ⚠️ Risk Assessment")
            risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            st.metric(
                "Risk Level",
                f"{risk_emoji.get(detection['risk_level'], '')} {detection['risk_level']}"
            )
            st.markdown(f"*{detection['risk_description']}*")

        st.markdown("---")

        # SECTION 4 & 5: Safe Navigation + Radar
        col_nav, col_radar = st.columns([1, 2])

        with col_nav:
            st.markdown("### 🧭 Safe Navigation")
            st.metric("Safe Direction", f"{detection['safe_angle']}°")
            st.metric("Cardinal", detection["safe_cardinal"])
            st.info(detection["safe_recommendation"])

            # Confidence gauge
            gauge_fig = create_confidence_gauge(
                detection["confidence"], detection["prediction"]
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_radar:
            st.markdown("### 📡 Radar Display")
            radar_fig = create_radar_plot(
                detection["distance"],
                detection["direction"],
                detection["safe_angle"]
            )
            st.plotly_chart(radar_fig, use_container_width=True)

    # ----------------------------------------------------------
    # SECTION 6: DATA VISUALIZATION & EDA
    # ----------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Data Analysis & Visualization")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Class Distribution",
        "🔥 Feature Importance",
        "📉 Signal Analysis",
        "🧮 Model Performance"
    ])

    with tab1:
        col_pie, col_count = st.columns(2)
        with col_pie:
            pie_fig = create_class_distribution_pie(df)
            st.plotly_chart(pie_fig, use_container_width=True)
        with col_count:
            count_fig = create_class_count_plot(df)
            st.plotly_chart(count_fig, use_container_width=True)

    with tab2:
        importance = detector.get_feature_importance()
        importance_fig = create_feature_importance_chart(importance)
        st.plotly_chart(importance_fig, use_container_width=True)

    with tab3:
        signal_fig = create_signal_strength_plot(df)
        st.plotly_chart(signal_fig, use_container_width=True)

        corr_fig = create_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)

        dist_fig = create_feature_distribution(df)
        st.plotly_chart(dist_fig, use_container_width=True)

    with tab4:
        metadata_path = os.path.join("model", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            metrics = metadata.get("metrics", {})
            cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

            col_cm, col_metrics = st.columns(2)
            with col_cm:
                cm_fig = create_confusion_matrix_chart(cm)
                st.plotly_chart(cm_fig, use_container_width=True)
            with col_metrics:
                st.markdown("### Model Performance Metrics")
                st.markdown(f"""
                | Metric | Score |
                |--------|-------|
                | Accuracy | {metrics.get('accuracy', 0) * 100:.2f}% |
                | Precision | {metrics.get('precision', 0) * 100:.2f}% |
                | Recall | {metrics.get('recall', 0) * 100:.2f}% |
                | F1 Score | {metrics.get('f1_score', 0) * 100:.2f}% |
                """)
                st.markdown(f"""
                **Model Configuration:**
                - Algorithm: Random Forest Classifier
                - Estimators: 100
                - Random State: 42
                - Features: 60 sonar frequency bands
                """)

    # ----------------------------------------------------------
    # SECTION 7: DETECTION HISTORY
    # ----------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📋 Detection History")

    if st.session_state.detection_history:
        history_data = []
        for d in st.session_state.detection_history[:20]:  # Show last 20
            history_data.append({
                "Timestamp": d["timestamp"],
                "Prediction": d["prediction"],
                "Confidence": f"{d['confidence']}%",
                "Risk": d["risk_level"],
                "Distance": f"{d['distance']}m",
                "Direction": f"{d['direction']}°",
                "Safe Dir": f"{d['safe_angle']}°"
            })

        history_df = pd.DataFrame(history_data)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()
    else:
        st.info("No detections yet. Run a detection to see history.")

    # ----------------------------------------------------------
    # FOOTER
    # ----------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #555;'>"
        "AI-Based Underwater Mine Detection System v1.0 | "
        "Random Forest ML Engine | Defense Intelligence Platform"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
