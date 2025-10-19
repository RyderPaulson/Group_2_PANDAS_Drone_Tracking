import torch
import cv2
import time
import statistics as stat

from CoTrackerCORE import CoTrackerCORE
from DetectionSystem import GroundingDINOCORE, find_sensor
from utils import prediction_in_box, send_coord, VideoDisplayer, LiveVideoViewer, scale_coord, preprocess_frame


DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def main(camera_id, text_prompt="personal drone", send_to_board=False, print_coord=False,
         write_out=False, disp_out=False, test_name="default",
         benchmarking=False) -> None:

    # Config variables
    window_size = 16
    rst_interval = window_size * 16 # How long to run cotracker before resetting it
    bb_check_interval = window_size * 8
    is_first_step = True

    # Instantiate different tracking objects
    cotracker = CoTrackerCORE(window_size)
    gd = GroundingDINOCORE(text_prompt)

    # Start video capture
    capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Create object to save output
    if write_out:
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        visualizer = VideoDisplayer("media/t" + test_name, fps, width, height)

    if disp_out:
        viewer = LiveVideoViewer()

    if benchmarking:
        t_bb = []
        t_cotracker = []

    frames_since_rst = 0
    sensor_coord = None

    while True:
        # Capture new frame
        ret, frame = capture.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error: Could not read frame. Exiting...")
            break

        frame_tensor_norm, frame_tensor, scale = preprocess_frame(frame)

        if frames_since_rst >= rst_interval or is_first_step:
            # Force reset state

            # Run full tracking workflow
            if benchmarking:
                start = time.perf_counter()

            box, _, _ = gd.detect(frame_tensor_norm)
            query_point = find_sensor(frame_tensor, box)
            cotracker.soft_rst(query_point)

            if benchmarking:
                end = time.perf_counter()
                t_bb.append(end-start)

            frames_since_rst = 0
            is_first_step = False

        elif frames_since_rst % bb_check_interval == 0:
            # Check tracked point still in drone
            box, _, _ = gd.detect(frame_tensor_norm)

            # Check that the latest prediction is still in the box
            reset_qp = prediction_in_box(sensor_coord, box)

            if not reset_qp:
                # Reset with new query_point
                query_point = find_sensor(frame_tensor, box)
                cotracker.hard_rst(query_point)

                frames_since_rst = 0

        if benchmarking:
            start = time.perf_counter()

        # Run CoTracker as normal
        sensor_coord, _ = cotracker.run_tracker(frame_tensor)
        if sensor_coord is None or sensor_coord.shape == torch.Size([1]):
            sensor_coord = None
        else:
            sensor_coord = scale_coord(sensor_coord[-1, -1, -1].tolist(), scale)

        if benchmarking:
            end = time.perf_counter()
            t_cotracker.append(end-start)

        frames_since_rst += 1

        # -- IO Options --
        if print_coord:
            if sensor_coord is None:
                print("No Point")
            else:
                print(f"x: {sensor_coord[0]:.2f} | y: {sensor_coord[1]:.2f}")

        if send_to_board:
            send_coord(sensor_coord)

        if write_out:
            visualizer.add_frame(sensor_coord, frame)

        if disp_out:
            viewer.show_frame(sensor_coord, frame)

    if benchmarking:
        print(f"Average time to find bounding box: {stat.fmean(t_bb):.5f}")
        print(f"Average time to run CoTracker: {max(t_cotracker):.5f}")

    if write_out:
        visualizer.save_video()

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_id = "media/ds_pan_cut.mp4"
    main(camera_id, write_out=True, print_coord=True, disp_out=True, benchmarking=True)
