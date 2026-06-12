#  AI-Based Underwater Mine Detection and Analysis System

A defense-style intelligent decision support system that uses machine learning to analyze sonar signal data and classify underwater objects as **Mine** or **Rock**. Built with Python, scikit-learn, and Streamlit.

---

## 📋 Overview

This system simulates the intelligence and analysis layer of a real-world underwater mine detection system. It processes 60-band sonar frequency data through a trained Random Forest classifier to provide:

- Object classification (Mine vs Rock)
- Confidence scoring
- Risk level assessment
- Direction and distance estimation
- Safe navigation recommendations
- Real-time radar visualization
- Audio/visual alert system

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **ML Classification** | Random Forest model trained on UCI Sonar dataset |
| **Confidence Scoring** | Probability-based confidence using `predict_proba()` |
| **Risk Analysis** | Three-tier risk system (LOW / MEDIUM / HIGH) |
| **Direction Estimation** | Signal-derived angular position (0°–180°) |
| **Distance Estimation** | Signal strength-based distance (10m–100m) |
| **Safe Navigation** | Automated course correction recommendations |
| **Radar Display** | Interactive polar radar visualization |
| **Alert System** | Visual + audio alerts for high-risk detections |
| **EDA Dashboard** | Correlation heatmaps, distributions, feature importance |
| **Detection History** | Timestamped log of all detections |

---

## 🏗️ Architecture

```
Dataset (sonar.csv)
    ↓
Data Preprocessing (label encoding, scaling)
    ↓
Random Forest Model Training (100 estimators)
    ↓
Prediction Engine (predict_proba)
    ↓
Confidence Score Generation
    ↓
Risk Analysis System (LOW/MEDIUM/HIGH)
    ↓
Direction & Distance Estimation
    ↓
Safe Navigation Recommendation
    ↓
Alert System (visual + audio)
    ↓
Streamlit Dashboard Visualization
```

---

## 📁 Project Structure

```
project/
├── data/
│   ├── sonar.csv              # UCI Sonar dataset
│   └── download_dataset.py    # Dataset download script
├── model/
│   ├── train_model.py         # Model training pipeline
│   ├── predict.py             # Prediction engine
│   ├── model.pkl              # Trained model (generated)
│   ├── scaler.pkl             # Feature scaler (generated)
│   └── metadata.json          # Model metadata (generated)
├── visualization/
│   ├── charts.py              # Pie charts, gauges, importance plots
│   ├── radar.py               # Radar-style visualization
│   └── eda.py                 # Exploratory data analysis plots
├── utils/
│   ├── risk_analysis.py       # Risk level calculation
│   ├── direction.py           # Direction/distance estimation
│   └── alerts.py              # Alert system configuration
├── assets/
│   ├── buzzer.wav             # Alarm audio (generated)
│   └── generate_buzzer.py     # Audio generation script
├── app.py                     # Main Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository - Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))

| Property | Value |
|----------|-------|
| Samples | 208 |
| Features | 60 (sonar frequency bands) |
| Classes | 2 (Mine=M, Rock=R) |
| Feature Range | 0.0 – 1.0 (normalized energy) |

Each feature represents the energy within a particular frequency band, integrated over time.

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd underwater-mine-detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset:**
```bash
python data/download_dataset.py
```

4. **Generate buzzer audio (optional):**
```bash
python assets/generate_buzzer.py
```

5. **Train the model:**
```bash
python model/train_model.py
```

6. **Launch the dashboard:**
```bash
streamlit run app.py
```

---

## 🖥️ Dashboard Guide

### Control Panel (Sidebar)
- Dataset statistics
- Detection mode selection (Random Sample / Upload CSV / Manual Input)
- Model performance metrics

### Detection Results
- Object classification with confidence percentage
- Direction angle and distance estimation
- Risk level with color-coded indicator

### Radar Display
- Ship position at center
- Mine position plotted using angle + distance
- Safe navigation direction arrow
- Range rings (25m, 50m, 75m, 100m)

### Data Analysis Tabs
- **Class Distribution:** Pie chart and bar chart
- **Feature Importance:** Top 15 Random Forest features
- **Signal Analysis:** Mean signal strength curves, correlation heatmap
- **Model Performance:** Confusion matrix and metrics table

### Detection History
- Timestamped table of all detections in current session

---

## 🤖 Random Forest Model

### Configuration
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1
)
```

### How It Works
1. **Ensemble Learning:** 100 decision trees vote on classification
2. **Feature Bagging:** Each tree uses a random subset of features
3. **Probability Output:** `predict_proba()` provides confidence scores
4. **Feature Importance:** Gini importance ranks the 60 frequency bands

### Evaluation Metrics
The model is evaluated on a 80/20 stratified train-test split using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🔮 Future Enhancements

- **Real Sonar Integration** — Connect to actual sonar hardware for live signal processing
- **Real-Time Streaming** — Process continuous sonar data streams
- **IoT Integration** — Connect with underwater sensor networks
- **Autonomous Navigation** — Automated course correction for unmanned vehicles
- **Multi-Sensor Fusion** — Combine sonar with magnetic and acoustic sensors
- **Deep Learning Upgrades** — CNN/LSTM models for temporal signal patterns
- **3D Visualization** — Volumetric underwater terrain mapping
- **Fleet Coordination** — Multi-vessel threat sharing network
- **Historical Pattern Analysis** — Mine field mapping over time

---

## 📄 License

This project is for educational and research purposes.

---

##  Acknowledgments

- UCI Machine Learning Repository for the Sonar dataset
- scikit-learn for the Random Forest implementation
- Streamlit for the dashboard framework
- Plotly for interactive visualizations
