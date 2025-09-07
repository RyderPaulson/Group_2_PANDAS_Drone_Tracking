import unittest
import os
import imageio.v3 as iio
from CoTrackerCORE import CoTrackerCORE

class test_CoTrackerCORE(unittest.TestCase):
    def test_apple_video(self):
        query_point  = [100, 100]
        cotracker = CoTrackerCORE(query_point)

        # Get video
        video_path = "./media/apple.mp4"

        if not os.path.isfile(video_path):
            self.fail("Video file does not exist")

        for i, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
            cotracker.run_tracker(frame)


if __name__ == '__main__':
    unittest.main()
