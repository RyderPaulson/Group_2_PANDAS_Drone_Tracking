import unittest
import os
import numpy as np
import imageio.v3 as iio
from IPython.display import Video as IPyVideo
import torch

# Import CoTracker Utilities
from CoTrackerCORE import CoTrackerCORE
from cotracker.utils.visualizer import Visualizer, read_video_from_path

# Import GUI for selecting points from Bishoy
from custom_point_tracker import select_points

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

class test_CoTrackerCORE(unittest.TestCase):
    def test_streamed_apple(self):
        query_point = [100, 100]

        # Get video
        video_path = "./media/apple.mp4"

        cotracker = CoTrackerCORE(query_point)
        if not os.path.isfile(video_path):
            self.fail("Video file does not exist")

        video_frames = []

        for i, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
            video_frames.append(frame)
            if i < cotracker.model.step * 2:
                cotracker.run_tracker(frame)
            else:
                pred_tracks, pred_visibility = cotracker.run_tracker(frame)

        video = torch.tensor(np.stack(video_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]

        vis = Visualizer(save_dir="./media", pad_value=120, linewidth=3)

        vis.visualize(video, pred_tracks, pred_visibility)

    def track_drone_from_nuav(self):
        # Get video
        video_path = "./media/panning_camera_mini_drone.mp4"
        query_point = [838, 392]
        query_frame = 52

        cotracker = CoTrackerCORE(query_point, query_frame=query_frame)

        if not os.path.isfile(video_path):
            self.fail("Video file does not exist")

        video_frames = []

        for i, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
            video_frames.append(frame)
            if i < cotracker.model.step * 2:
                cotracker.run_tracker(frame)
            else:
                pred_tracks, pred_visibility = cotracker.run_tracker(frame)

            if i > 400:
                break

        video = torch.tensor(np.stack(video_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]

        vis = Visualizer(save_dir="./media", pad_value=120, linewidth=3)

        vis.visualize(video, pred_tracks, pred_visibility)

if __name__ == "__main__":
    unittest.main()
