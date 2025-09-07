import unittest
import cotracker_matlab_interface as cmi
import cv2
import numpy as np

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

class test_matlab_interfacce(unittest.TestCase):
    def test_initialization(self):
        initialization_status = cmi.initialize_tracker()

        self.assertTrue(initialization_status)

    def test_image_annotation(self):
        # Video file path
        video_path = './media/apple.mp4'

        # Open video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.fail("Could not open video file {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Initialize tracker - set POI at center of frame initially
        cmi.initialize_tracker(poi_x=width // 2, poi_y=height // 2)

        frame_count = 0
        while frame_count < total_frames:
            # Read frame from video
            ret, bgr_frame = cap.read()

            if not ret:
                self.fail("End of video or failed to read frame")

            # Convert BGR (OpenCV) to RGB (CoTracker expects RGB)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # Process frame with CoTracker
            annotated_frame = cmi.process_single_frame(rgb_frame)

            # Get tracking info
            pos, vis = cmi.get_poi_position()
            if pos is not None:
                print(f"Frame {frame_count}: POI at ({pos[0]:.1f}, {pos[1]:.1f}), visibility: {vis:.2f}")
            else:
                print(f"Frame {frame_count}: No POI position available")

            # Convert back to BGR for display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            # Show the frame
            cv2.imshow('CoTracker POI Tracking', display_frame)

            # Control playback
            key = cv2.waitKey(int(1000 / fps)) & 0xFF  # Wait time based on original FPS
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar to pause
                cv2.waitKey(0)
            elif key == ord('r'):  # 'r' to reset tracker
                cmi.reset_tracker()
                print("Tracker reset")

            frame_count += 1

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
