import torch
import numpy as np

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Global tracker instance for MATLAB interface
_global_tracker = None


def initialize_tracker(checkpoint_path=None, poi_x=None, poi_y=None):
    global _global_tracker
    try:
        _global_tracker = CoTrackerMATLAB(
            checkpoint=checkpoint_path,
            poi_x=poi_x,
            poi_y=poi_y
        )
        return True
    except Exception as e:
        print(f"Failed to initialize tracker: {e}")
        return False


def set_poi(x, y):
    global _global_tracker
    if _global_tracker is None:
        print("Tracker not initialized. Call initialize_tracker() first.")
        return False

    try:
        _global_tracker.set_poi(x, y)
        return True
    except Exception as e:
        print(f"Failed to set POI: {e}")
        return False


def process_single_frame(frame):
    global _global_tracker
    if _global_tracker is None:
        raise RuntimeError("Tracker not initialized. Call initialize_tracker() first.")

    return _global_tracker.process_frame(frame)


def get_poi_position():
    global _global_tracker
    if _global_tracker is None:
        return None, None

    return _global_tracker.get_current_poi_position()


def reset_tracker():
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()