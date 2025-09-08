# Module providing an interface for connecting CoTracker with MATLAB
import torch
import numpy as np

# Global instance of the tracking
_global_tracker = None

def initialize(query_point, query_frame=0, window_size=10, checkpoint_path=None):
    """

    :param query_point: -> Coordinates for the point to track.
    :param query_frame: -> The frame number at which point tracking begins. By default is set to 0.
    :param window_size: -> The size of the sliding window that co-tracker will use. By default is 10.
    :param checkpoint_path: -> The path to the model checkpoint. By default loads a new model.
    :return: status: -> The initialization status of the tracker as a boolean.
    """
    pass

def run(new_frame):
    """
    General function for running through a provided video and returning an annotated frame.
    :param new_frame:
    :return: annotated_frame:
    """
    pass

def get_query_position():
    pass