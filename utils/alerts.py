"""
Alert System for underwater mine detection.
Handles visual and audio alerts based on risk level.
"""
import os


def get_alert_config(risk_level):
    """
    Get alert configuration based on risk level.
    
    Args:
        risk_level: str, "LOW", "MEDIUM", or "HIGH"
    
    Returns:
        dict with alert configuration
    """
    alerts = {
        "HIGH": {
            "trigger_audio": True,
            "flash_warning": True,
            "border_color": "#FF0000",
            "background_color": "#330000",
            "icon": "🚨",
            "title": "⚠️ EMERGENCY ALERT ⚠️",
            "message": "MINE DETECTED WITH HIGH CONFIDENCE!\nImmediate course correction required!",
            "audio_file": "assets/buzzer.wav",
            "css_class": "alert-high"
        },
        "MEDIUM": {
            "trigger_audio": False,
            "flash_warning": False,
            "border_color": "#FFA500",
            "background_color": "#332200",
            "icon": "⚠️",
            "title": "CAUTION",
            "message": "Possible mine detected. Proceed with caution.",
            "audio_file": None,
            "css_class": "alert-medium"
        },
        "LOW": {
            "trigger_audio": False,
            "flash_warning": False,
            "border_color": "#00FF00",
            "background_color": "#003300",
            "icon": "✅",
            "title": "ALL CLEAR",
            "message": "No immediate threat detected.",
            "audio_file": None,
            "css_class": "alert-low"
        }
    }
    return alerts.get(risk_level, alerts["LOW"])


def generate_alert_html(alert_config):
    """
    Generate HTML for alert display in Streamlit.
    
    Args:
        alert_config: dict from get_alert_config()
    
    Returns:
        str: HTML string for alert display
    """
    html = f"""
    <div style="
        border: 2px solid {alert_config['border_color']};
        background-color: {alert_config['background_color']};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    ">
        <h2 style="color: {alert_config['border_color']}; margin: 0;">
            {alert_config['icon']} {alert_config['title']} {alert_config['icon']}
        </h2>
        <p style="color: white; font-size: 16px; margin-top: 10px;">
            {alert_config['message']}
        </p>
    </div>
    """
    return html


def get_buzzer_path():
    """Get path to buzzer audio file."""
    return os.path.join(os.path.dirname(__file__), "..", "assets", "buzzer.wav")
