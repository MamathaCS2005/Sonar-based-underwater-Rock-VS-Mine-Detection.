"""
Risk Analysis System for underwater mine detection.
Generates risk levels based on prediction confidence.
"""


def calculate_risk_level(confidence, prediction):
    """
    Calculate risk level based on confidence and prediction.
    
    Args:
        confidence: float, prediction confidence percentage (0-100)
        prediction: str, "Mine" or "Rock"
    
    Returns:
        dict with risk level, color, and description
    """
    # Only mines pose a risk
    if prediction == "Rock":
        return {
            "level": "LOW",
            "color": "green",
            "emoji": "🟢",
            "description": "Object classified as rock. No threat detected.",
            "action": "Continue current navigation path."
        }
    
    # Mine detected - assess risk based on confidence
    if confidence > 85:
        return {
            "level": "HIGH",
            "color": "red",
            "emoji": "🔴",
            "description": "HIGH CONFIDENCE MINE DETECTION. Immediate action required!",
            "action": "EMERGENCY: Change course immediately. Alert all personnel.",
            "alert": True
        }
    elif confidence >= 60:
        return {
            "level": "MEDIUM",
            "color": "orange",
            "emoji": "🟡",
            "description": "Possible mine detected. Proceed with caution.",
            "action": "Reduce speed. Prepare for course correction.",
            "alert": False
        }
    else:
        return {
            "level": "LOW",
            "color": "yellow",
            "emoji": "🟢",
            "description": "Low confidence detection. Monitor closely.",
            "action": "Continue monitoring. Increase sensor sensitivity.",
            "alert": False
        }
