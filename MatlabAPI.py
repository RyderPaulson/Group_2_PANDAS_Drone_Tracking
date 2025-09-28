# Module providing an interface for connecting CoTracker with MATLAB
import os

from CoTrackerCORE import CoTrackerCORE as CoTracker


class MatlabAPI:

    def __init__(
        self,
        query_point: list,
        query_frame: int = 0,
        window_size: int = 10,
        checkpoint_path: str = None,
    ):
        """
        Initializes the CoTrackerAPI object.
        :param query_point: -> Coordinates for the point to track.
        :param query_frame: -> The frame number at which point tracking begins. By default is set to 0.
        :param window_size: -> The size of the sliding window that co-tracker will use. By default is 10.
        :param checkpoint_path: -> The path to the model checkpoint. By default loads a new model.
        :return self: -> API Object.
        """
        # Raise error if query point is not sent as a 2 element list.
        if not (isinstance(query_point, list) and len(query_point) == 2):
            raise TypeError("The query point must be a list of length 2.")

        # Raise error if query frame is not an integer
        if not (isinstance(query_frame, int) and query_frame >= 0):
            raise TypeError(
                "The query frame must be an integer greater than or equal to 0."
            )

        # Raise error if window size is not an integer greater than or equal to 1
        # TODO: Does Cotracker have a smaller minimum window size?
        if not (isinstance(window_size, int) and window_size >= 1):
            raise TypeError(
                "The sliding window size must be an integer greater than or equal to 1."
            )

        # Raise error if checkpoint path is not NULL or a String
        if not (isinstance(checkpoint_path, str) or isinstance(checkpoint_path, None)):
            raise TypeError("The checkpoint path must either be NULL or a string.")

        # Raise error if path to checkpoint is non-existant

        if (
            not isinstance(checkpoint_path, None)
            and os.path.exists(checkpoint_path) is False
        ):
            raise ValueError("No file exists on the checkpoint path.")

        # Instantiate CoTracker model
        self.cotracker = CoTracker(
            query_point, query_frame, window_size, checkpoint_path
        )


def run(new_frame):
    """
    General function for running through a provided video and returning an annotated frame.
    Converting video to float will be handled in the core CoTracker module.
    :param new_frame:
    :return: annotated_frame:
    """
    pass


def get_query_position():
    pass
