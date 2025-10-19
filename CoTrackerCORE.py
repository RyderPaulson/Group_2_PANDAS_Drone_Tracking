import torch
import numpy as np

from cotracker.predictor import CoTrackerPredictor


DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

class CoTrackerCORE:
    """
    Class for the core CoTracker functions. It is engineered with the
    intentions to be fully separated of unique MATLAB requirements.
    """

    # ----------------------- Public Methods -----------------------

    def __init__(self, window_size=16, checkpoint_path=None):
        """

        :param query_point: -> The point to track in [x, y] format.
        :param query_frame: -> The frame used for querying CoTracker.
        :param checkpoint_path: -> The path for the checkpoint to load the model from.
                                   Default: None which loads a fresh model from Torch Hub.
        """
        # If path provided, load it. Otherwise, load fresh model
        self.checkpoint_path = checkpoint_path

        self.pred_tracks = torch.tensor([0])
        self.pred_visibility = torch.tensor([0])

        # Declaration of vars set in initialization
        self.is_first_step = None
        self.model = None
        self.query = None
        self.window_frames = list()
        self.window_size = window_size

    def run_tracker(self, new_frame: torch.tensor):
        """
        Collect a new frame and if the frame buffer is full, process them.
        :param new_frame:
        :return pred_tracks, pred_visibility: -> the predictions and visibility of tracked points.
        """
        self.window_frames.append(
            new_frame
            .to(DEFAULT_DEVICE)
        )

        num_frames = len(self.window_frames)

        # Dump old frames. The program should hold 3x the model step frames in memory at a time.
        if num_frames >= self.model.step * 3:
            self.window_frames = self.window_frames[self.model.step:]

        if num_frames % self.model.step == 0 and num_frames != 0:
            self.pred_tracks, self.pred_visibility = self._process_step()
            self.is_first_step = False

        if num_frames >= self.model.step * 2 and self.pred_tracks is None:
            self.pred_tracks = torch.tensor([0])

        return self.pred_tracks, self.pred_visibility

    def hard_rst(self, query_point, query_frame=0):
        self._initialize_tracker(query_point, query_frame)

        # Frame buffer for windowed processing
        self.window_frames = list()

    def soft_rst(self, query_point, query_frame=0):
        self._initialize_tracker(query_point, query_frame)

    # ----------------------- Private Methods -----------------------

    def _process_step(self):
        """Process all buffered frames and return tracks and visibility."""
        video_chunk = torch.stack(self.window_frames[-self.model.step * 2 :])[None] # (1, T, 3, H, W)
        return self.model(video_chunk, is_first_step=self.is_first_step, queries=self.query[None])

    def _initialize_tracker(self, query_point, query_frame) -> None:
        # Initialize the model
        if self.checkpoint_path is None:
            self.model = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_online"
            ).to(DEFAULT_DEVICE)
        else:
            self.model = (CoTrackerPredictor(checkpoint=self.checkpoint_path)
                            .to(DEFAULT_DEVICE)
                          )

        # self.model.step = self.window_size//2

        # Put query into T, H, W format and move to GPU if available
        self.query = torch.tensor(
            [
                [query_frame, query_point[0], query_point[1]],
            ]
        ).float()
        if torch.cuda.is_available():
            self.query = self.query.cuda()

        self.is_first_step = True

        initialization_status = (
            f"CoTracker initialized on {DEFAULT_DEVICE}\n"
            f"Model step size: {self.model.step}"
        )

        print(initialization_status)
