import torch
import cv2
import argparse

from CoTrackerCORE import CoTrackerCORE
from DetectionSystem import GroundingDINOCORE, find_sensor
import utils


DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def main(camera_id,
         text_prompt="drone",
         send_to_board=False,
         print_coord=False,
         write_out=False,
         disp_out=False,
         benchmarking=False,
         test_name="default",
         window_size=16,
         rst_interval_mult=16,
         bb_check_mult=8,
         max_img_size=800,
         target_color=[128, 128, 128]) -> None:

    # Config variables
    rst_interval = window_size * rst_interval_mult
    bb_check_interval = window_size * bb_check_mult
    is_first_step = True

    # Instantiate different tracking objects
    cotracker = CoTrackerCORE(window_size)
    gd = GroundingDINOCORE(text_prompt)

    # Start video capture
    capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        print("Error: Could not open camera.")
        exit()

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    io_options = utils.IOOptions(test_name,
                                 send_to_board,
                                 print_coord,
                                 write_out,
                                 disp_out,
                                 benchmarking,
                                 capture)

    frames_since_rst = 0
    sensor_coord = None

    while True:
        # Capture new frame
        ret, frame = capture.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error: Could not read frame. Exiting...")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get preprocessed frames
        frame_tensor_norm, frame_tensor, scale = utils.preprocess_frame(
            frame_rgb, max_img_size
        )

        if frames_since_rst >= rst_interval or is_first_step:
            # Force reset state
            # Run full tracking workflow
            box, _, _ = gd.detect(frame_tensor)
            query_point = find_sensor(frame_tensor, box, target_color)
            cotracker.soft_rst(query_point)

            frames_since_rst = 0
            is_first_step = False

        elif frames_since_rst % bb_check_interval == 0:
            # Check tracked point still in drone
            # IMPORTANT: Pass frame_tensor (0-255 range) to GroundingDINO
            box, _, _ = gd.detect(frame_tensor)

            # Check that the latest prediction is still in the box
            reset_qp = utils.prediction_in_box(sensor_coord, box)

            if not reset_qp:
                # Reset with new query_point
                query_point = find_sensor(frame_tensor, box, target_color)
                cotracker.hard_rst(query_point)

                frames_since_rst = 0

        sensor_coord, _ = cotracker.run_tracker(
            frame_tensor_norm
        )
        if sensor_coord is None or sensor_coord.shape == torch.Size([1]):
            sensor_coord = None
            normalized_coord = None
        else:
            scaled_coord = sensor_coord[-1, -1, -1].tolist()
            sensor_coord = utils.scale_coord(scaled_coord, scale)

            normalized_coord = normalize_coord(scaled_coord, width, height)

        frames_since_rst += 1

        io_options.run(sensor_coord, normalized_coord, frame)

    del io_options # Call destructor which will print results

    capture.release()
    cv2.destroyAllWindows()

def camera_id_type(value):
    """Convert camera_id to int if possible, otherwise keep as string."""
    try:
        return int(value)
    except ValueError:
        return value

def normalize_coord(sensor_coord, w, h):
    normalized_coord = [sensor_coord[0]/w,
                        sensor_coord[1]/h,
                        ]

    normalized_coord[0] = normalized_coord[0] - 0.5
    normalized_coord[1] = -1 * normalized_coord[1] + 0.5

    return normalized_coord


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drone tracking system using CoTracker and Grounding DINO"
    )

    # Required arguments
    parser.add_argument(
        "camera_id", type=camera_id_type, help="Camera ID (integer) or path to video file"
    )

    # Optional arguments
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="drone",
        help="Text prompt for object detection (default: 'drone')",
    )

    parser.add_argument(
        "--send-to-board", action="store_true", help="Send coordinates to board"
    )

    parser.add_argument(
        "--print-coord", action="store_true", help="Print coordinates to console"
    )

    parser.add_argument("--write-out", action="store_true", help="Write output to file")

    parser.add_argument("--disp-out", action="store_true", help="Display output window")

    parser.add_argument(
        "--benchmarking", action="store_true", help="Enable benchmarking mode"
    )

    parser.add_argument(
        "--test-name",
        type=str,
        default="default",
        help="Name for the test run (default: 'default')",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="CoTracker window size (default: 16)",
    )

    parser.add_argument(
        "--rst-interval-mult",
        type=int,
        default=16,
        help="Reset interval multiplier (default: 16, rst_interval = window_size * mult)",
    )

    parser.add_argument(
        "--bb-check-mult",
        type=int,
        default=8,
        help="Bounding box check multiplier (default: 8, bb_check_interval = window_size * mult)",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=1330,
        help="The max width that the image will be transformed down into.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        camera_id=args.camera_id,
        text_prompt=args.text_prompt,
        send_to_board=args.send_to_board,
        print_coord=args.print_coord,
        write_out=args.write_out,
        disp_out=args.disp_out,
        benchmarking=args.benchmarking,
        test_name=args.test_name,
        window_size=args.window_size,
        rst_interval_mult=args.rst_interval_mult,
        bb_check_mult=args.bb_check_mult,
        max_img_size=args.img_size
    )
