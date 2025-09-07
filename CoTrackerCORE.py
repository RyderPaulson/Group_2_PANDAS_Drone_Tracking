import torch
import numpy as np
import os

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

class CoTrackerCORE:
    """
    Class for the core CoTracker functions. It is engineered with the
    intentions to be fully separated of unique MATLAB requirements.
    """

    # ----------------------- Public Methods -----------------------

    def __init__(self, query_point,  query_frame=0, window_size=10):
        # Always load a fresh instance of the model
        checkpoint_path = os.path.join(".","co-tracker","checkpoints","baseline_online.pth")
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path)

        # Put query into [T, X, Y] format where T is time
        self.query_point = torch.tensor([
            [query_frame, query_point[0], query_point[1]],
        ])
        if torch.cuda.is_available():
            self.query_point = self.query_point.cuda()
        self.query_frame = query_frame

        # Frame buffer for windowed processing
        self.frame_count = 0
        self.window_size = window_size
        self.window_frames = list()

        # Property for most recently visualized frame
        self.most_recent_frame = None

        self.is_first_step = True

        initialization_status = (f"CoTracker initialized on {DEFAULT_DEVICE}"
                                 f"Model step size: {self.model.step}")

        print(initialization_status)

    def run_tracker(self, new_frame):
        """
        Collect a new frame and if the frame buffer is full, process them.
        :param new_frame:
        :return: NULL or annotated frame
        """
        self.frame_count += 1

        self.window_frames.append(new_frame)

        if self.frame_count % self.model.step == 0 and self.frame_count != self.model.step and self.frame_count != 0:
            try:
                # Process set of frames
                pred_tracks, pred_visibility = self._process_step()

                self.is_first_step = False

                # Visualize set of frames
                self._visualize_frame(pred_tracks, pred_visibility)

                return self.most_recent_frame
            except Exception as e:
                print(f"Frame Count: {self.frame_count}")
                raise e

        else:
            # Return most recent visualized frame. This will make it so that the literal frame rate of the output video
            # stream is the same as the input one but the effective frame rate will be equivalent to fps/model.step.
            #return self.most_recent_frame
            pass

    # ----------------------- Private Methods -----------------------

    def _process_step(self):
        """Process all buffered frames and return tracks and visibility."""
        video_chunk = (torch.tensor(np.stack(self.window_frames[-self.model.step * 2:]), device=DEFAULT_DEVICE)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return self.model(
            video_chunk,
            queries=self.query_point[None],
        )


    def _visualize_frame(self, tracks, visibility):
        """Use the cotracker visualization suite to produce an annotated frame."""

        video = (torch.tensor(np.stack(self.window_frames[-self.model.step * 2:]), device=DEFAULT_DEVICE)
        .float()
        .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        vis = Visualizer(
            save_dir='./media',
            linewidth=6,
            mode='cool',
            tracks_leave_trace=-1
        )
        vis.visualize(
            video=video,
            tracks=tracks,
            visibility=visibility,
            filename=f'queries{self.frame_count}')