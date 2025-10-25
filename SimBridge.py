import torch
import numpy as np
import cv2

from CoTrackerCORE import CoTrackerCORE
from DetectionSystem import GroundingDINOCORE, find_sensor
import utils


DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class TrackingBridge:
    """
    Bridge between MATLAB/Simulink simulation and Python tracking system.
    Processes individual frames from Simulink and returns annotated results.
    """

    def __init__(
        self,
        text_prompt="drone",
        window_size=16,
        rst_interval_mult=16,
        bb_check_mult=8,
        max_img_size=800,
    ):
        """
        Initialize the tracking bridge.

        Args:
            text_prompt: Text prompt for GroundingDINO detection
            window_size: CoTracker window size
            rst_interval_mult: Reset interval multiplier
            bb_check_mult: Bounding box check interval multiplier
            max_img_size: Maximum image dimension for processing
        """
        # Config variables
        self.window_size = window_size
        self.rst_interval = window_size * rst_interval_mult
        self.bb_check_interval = window_size * bb_check_mult
        self.max_img_size = max_img_size

        # Instantiate tracking objects
        self.cotracker = CoTrackerCORE(window_size)
        self.gd = GroundingDINOCORE(text_prompt)

        # State tracking
        self.frames_since_rst = 0
        self.is_first_step = True
        self.sensor_coord = None

        print(f"TrackingBridge initialized on {DEFAULT_DEVICE}")
        print(f"  Text prompt: {text_prompt}")
        print(f"  Window size: {window_size}")
        print(f"  Reset interval: {self.rst_interval}")
        print(f"  BB check interval: {self.bb_check_interval}")

    def process_frame(self, frame):
        """
        Process a single frame from Simulink.

        Args:
            frame: numpy array (H, W, C) with color values in [0, 255], BGR format from MATLAB

        Returns:
            Annotated frame as numpy array (H, W, C) in BGR format
        """
        # Convert from BGR (OpenCV/MATLAB) to RGB for processing
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame

        # Preprocess frame
        frame_tensor_norm, frame_tensor, scale = utils.preprocess_frame(
            frame_rgb, self.max_img_size
        )

        # Tracking logic
        if self.frames_since_rst >= self.rst_interval or self.is_first_step:
            # Force reset state - run full tracking workflow
            box, _, _ = self.gd.detect(frame_tensor_norm)
            query_point = find_sensor(frame_tensor, box)
            self.cotracker.soft_rst(query_point)

            self.frames_since_rst = 0
            self.is_first_step = False

        elif self.frames_since_rst % self.bb_check_interval == 0:
            # Check tracked point still in drone bounding box
            box, _, _ = self.gd.detect(frame_tensor_norm)

            # Check that the latest prediction is still in the box
            if self.sensor_coord is not None:
                reset_qp = utils.prediction_in_box(self.sensor_coord, box)

                if not reset_qp:
                    # Reset with new query_point
                    query_point = find_sensor(frame_tensor, box)
                    self.cotracker.hard_rst(query_point)
                    self.frames_since_rst = 0

        # Run CoTracker
        sensor_coord, _ = self.cotracker.run_tracker(frame_tensor)

        if sensor_coord is None or sensor_coord.shape == torch.Size([1]):
            self.sensor_coord = None
        else:
            # Extract and scale coordinates
            self.sensor_coord = utils.scale_coord(
                sensor_coord[-1, -1, -1].tolist(), scale
            )

        self.frames_since_rst += 1

        # Annotate the original frame
        annotated_frame = self._annotate_frame(frame_rgb, self.sensor_coord)

        # Convert back to BGR for MATLAB/OpenCV
        # annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        annotated_frame_bgr = annotated_frame

        print("Sensor Coord: [%i], [%i]", self.sensor_coord[0], self.sensor_coord[1])

        return annotated_frame_bgr

    def _annotate_frame(self, frame, coord):
        """
        Draw tracking point on frame.

        Args:
            frame: numpy array (H, W, C) in RGB format
            coord: [x, y] coordinates or None

        Returns:
            Annotated frame as numpy array
        """
        # Make a copy to avoid modifying original
        annotated = frame.copy()

        if coord is not None:
            # Draw green circle at tracked point
            cv2.circle(
                annotated,
                (int(coord[0]), int(coord[1])),
                radius=5,
                color=(0, 255, 0),  # Green in RGB
                thickness=-1,
            )

        return annotated

    def reset(self):
        """Reset the tracking system state."""
        self.frames_since_rst = 0
        self.is_first_step = True
        self.sensor_coord = None
        print("TrackingBridge reset")
