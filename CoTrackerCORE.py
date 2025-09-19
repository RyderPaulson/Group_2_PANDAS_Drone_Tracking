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

    def __init__(
        self, query_point, query_frame=0, checkpoint_path=None
    ):
        """

        :param query_point: -> Coordinates for the point to track. With the current implementation, it is only possible to
        track one point at a time.
        :param query_frame: -> The time coordinate for the query.
                               Default: 0
        :param checkpoint_path: -> The path for the checkpoint to load the model from.
                                   Default: None which loads a fresh model from Torch Hub.
        """
        # If path provided, load it. Otherwise, load fresh model
        if checkpoint_path is None:
            self.model = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_online"
            ).to(DEFAULT_DEVICE)
        else:
            self.model = CoTrackerPredictor(checkpoint=checkpoint_path).to(
                DEFAULT_DEVICE
            )

        # Put query into [T, X, Y] format where T is time
        self.query_point = torch.tensor(
            [
                [query_frame, query_point[0], query_point[1]],
            ]
        )
        if torch.cuda.is_available():
            self.query_point = self.query_point.cuda()
        self.query_frame = query_frame

        # Frame buffer for windowed processing
        self.frame_count = 0
        self.window_frames = []

        # Property for most recently visualized frame
        self.is_first_step = True
        self.pred_tracks = torch.tensor([0])
        self.pred_visibility = torch.tensor([0])

        initialization_status = (
            f"CoTracker initialized on {DEFAULT_DEVICE}"
            f"Model step size: {self.model.step}"
        )

        print(initialization_status)

    def run_tracker(self, new_frame):
        """
        Collect a new frame and if the frame buffer is full, process them.
        :param new_frame:
        :return [x, y] prediction location:
        """
        self.frame_count += 1

        self.window_frames.append(new_frame)

        # Dump old frames. The program should hold 3x the model step frames in memory at a time.
        if len(self.window_frames) >= self.model.step * 3:
            self.window_frames = self.window_frames[self.model.step:]

        if self.frame_count % self.model.step == 0 and self.frame_count != 0:
            self.pred_tracks, self.pred_visibility = self._process_step()
            self.is_first_step = False

        return self.pred_tracks, self.pred_visibility

    # ----------------------- Private Methods -----------------------

    def _process_step(self):
        """Process all buffered frames and return tracks and visibility."""
        video_chunk = (
            torch.tensor(
                np.stack(self.window_frames[-self.model.step * 2 :]),
                device=DEFAULT_DEVICE,
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        # TODO Get CoTracker to track a single point rather than a grid
        # return self.model(video_chunk, is_first_step=self.is_first_step, queries=self.query_point[None])
        return self.model(video_chunk, is_first_step=self.is_first_step, grid_size=30)
