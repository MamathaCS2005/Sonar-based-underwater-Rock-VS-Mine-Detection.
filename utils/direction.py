"""
Direction and Distance Estimation Module.
Generates simulated direction and distance based on sonar signal characteristics.
"""
import numpy as np


def estimate_direction(features):
    """
    Estimate direction angle of detected object.
    Uses signal characteristics to generate a logical direction.
    
    Args:
        features: numpy array of sonar features
    
    Returns:
        float: direction angle in degrees (0-180)
    """
    # Use a combination of features to derive a consistent direction
    # Higher frequency features (later columns) indicate angular position
    features = np.array(features)
    
    # Use weighted sum of features to generate angle
    weights = np.linspace(0.5, 1.5, len(features))
    weighted_sum = np.sum(features * weights)
    
    # Normalize to 0-180 degree range
    angle = (weighted_sum * 100) % 180
    return round(angle, 1)


def estimate_distance(features):
    """
    Estimate distance to detected object.
    Uses signal strength to approximate distance.
    
    Args:
        features: numpy array of sonar features
    
    Returns:
        float: estimated distance in meters (10-100)
    """
    features = np.array(features)
    
    # Signal strength inversely correlates with distance
    # Stronger signals = closer objects
    mean_signal = np.mean(features)
    
    # Map signal strength to distance (10m - 100m)
    # Higher signal = closer (lower distance)
    distance = 100 - (mean_signal * 90)
    distance = np.clip(distance, 10, 100)
    
    return round(float(distance), 1)


def calculate_safe_direction(mine_angle):
    """
    Calculate safe navigation direction away from detected mine.
    
    Args:
        mine_angle: float, angle where mine is detected (0-180)
    
    Returns:
        dict with safe direction info
    """
    # Safe direction is opposite to mine direction
    # Add 90-120 degrees offset for safety margin
    if mine_angle <= 90:
        safe_angle = mine_angle + 120
        if safe_angle > 180:
            safe_angle = 180
    else:
        safe_angle = mine_angle - 120
        if safe_angle < 0:
            safe_angle = 0
    
    # Determine cardinal direction
    if safe_angle < 30:
        cardinal = "Port (Left)"
    elif safe_angle < 60:
        cardinal = "Port-Forward"
    elif safe_angle < 120:
        cardinal = "Forward"
    elif safe_angle < 150:
        cardinal = "Starboard-Forward"
    else:
        cardinal = "Starboard (Right)"
    
    return {
        "safe_angle": round(safe_angle, 1),
        "cardinal_direction": cardinal,
        "recommendation": f"Navigate to {safe_angle}° ({cardinal}) to avoid detected object.",
        "mine_angle": mine_angle
    }


def get_coordinates(distance, angle_degrees):
    """
    Convert polar coordinates (distance, angle) to cartesian (x, y).
    
    Args:
        distance: float, distance in meters
        angle_degrees: float, angle in degrees
    
    Returns:
        tuple: (x, y) coordinates
    """
    angle_rad = np.radians(angle_degrees)
    x = distance * np.cos(angle_rad)
    y = distance * np.sin(angle_rad)
    return round(float(x), 2), round(float(y), 2)
