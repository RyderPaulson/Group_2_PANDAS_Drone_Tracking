# PyPi packages
import gc
import unittest
import os
import numpy as np
import imageio.v3 as iio
import torch

# From cloned repositories
from CoTrackerCORE import CoTrackerCORE
from cotracker.utils.visualizer import Visualizer, read_video_from_path

# Unique to project
from utils import print_frames_analyzed, preprocess_frame

from utils import LiveVideoViewer

# Set the default device, if an Nvidia GPU is present it will be used where possible
DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class TestCoTrackerCORE(unittest.TestCase):
    def test_ds_pan_cut(self):
        # Get video
        video_path = "../media/ds_pan_cut.mp4"
        video_frames = []
        for frame in iio.imiter(video_path):
            video_frames.append(frame)

        # Setup CoTracker
        query_point = [882, 386]
        query_frame = 0
        cotracker = CoTrackerCORE()
        cotracker.hard_rst(query_point, query_frame=query_frame)

        # Remove frames at the end of the video so that it is evenly divisible with the model step
        video_frames = video_frames[: -(len(video_frames) % cotracker.model.step)]

        if not os.path.isfile(video_path):
            self.fail("Video file does not exist")

        # Iterate through video
        for i, frame in enumerate(video_frames):
            frame, _ = preprocess_frame(frame)
            pred_tracks, pred_visibility = cotracker.run_tracker(frame)

            print_frames_analyzed(i, 25)

        video = torch.tensor(np.stack(video_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]

        del cotracker, video_frames
        gc.collect()

        # Visualize the predicted tracks.
        vis = Visualizer(save_dir="../media", pad_value=10, linewidth=3, mode="cool", fps=30)
        vis.visualize(video, pred_tracks, pred_visibility, filename="t_ds_pan_cut")

    def test_ds_pan_full(self):
        test_name = "t_ds_pan_full"
        src_name = "../media/ds_pan.mp4"

        # Get video
        video_path = "./media/" + src_name
        video_frames = []
        for frame in iio.imiter(video_path):
            video_frames.append(frame)

        # Setup CoTracker
        query_point = [882, 386]
        query_frame = 0
        cotracker = CoTrackerCORE()
        cotracker.hard_rst(query_point, query_frame=query_frame)

        # Remove frames at the end of the video so that it is evenly divisible with the model step
        video_frames = video_frames[: -(len(video_frames) % cotracker.model.step)]

        if not os.path.isfile(video_path):
            self.fail("Video file does not exist")

        # Iterate through video
        for i, frame in enumerate(video_frames):
            pred_tracks, pred_visibility = cotracker.run_tracker(frame)

            print_frames_analyzed(i, 25)

        video = torch.tensor(np.stack(video_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]

        del cotracker, video_frames
        gc.collect()

        # Visualize the predicted tracks.
        vis = Visualizer(save_dir="../media", pad_value=10, linewidth=3, mode="cool", fps=30)
        vis.visualize(video, pred_tracks, pred_visibility, filename=test_name)

    def test_ds_frame(self):
        test_name = "t_ds_frame"
        src_name = "../media/ds_frame.mp4"

        # Get video
        video_path = "./media/" + src_name
        video_frames = []
        for frame in iio.imiter(video_path):
            video_frames.append(frame)

        # Setup CoTracker
        query_point = [882, 386]
        query_frame = 0
        cotracker = CoTrackerCORE()
        cotracker.hard_rst(query_point, query_frame=query_frame)

        # Remove frames at the end of the video so that it is evenly divisible with the model step
        video_frames = video_frames[: -(len(video_frames) % cotracker.model.step)]

        if not os.path.isfile(video_path):
            self.fail("Video file does not exist")

        # Iterate through video
        for i, frame in enumerate(video_frames):
            pred_tracks, pred_visibility = cotracker.run_tracker(frame)

            print_frames_analyzed(i, 25)

        video = torch.tensor(np.stack(video_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]

        del cotracker, video_frames
        gc.collect()

        # Visualize the predicted tracks.
        vis = Visualizer(
            save_dir="../media", pad_value=10, linewidth=3, mode="cool", fps=30
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=test_name)


if __name__ == "__main__":
    unittest.main()
