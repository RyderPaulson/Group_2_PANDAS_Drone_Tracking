# PyPi packages
import torch
import cv2

try:
    from motorControl import trackCoords, servo
    _servoX = servo(32)
    _servoY = servo(33)
except:
    print("Not currently running on Jetson. Motorcontrols cannot be used.")

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def preprocess_frame(img_array, max_size):
    """
    img_array: numpy array (H, W, C) with color values in [0, 255]
    returns: torch tensor (C, H, W) normalized
    """
    # Resize (values are based on groundedDINO load_img function
    h, w = img_array.shape[:2]
    target_size = int((2/3)*max_size)

    # Calculate new size maintaining aspect ratio
    scale = target_size / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert to tensor
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor.to(DEFAULT_DEVICE)

    # Convert to tensor and normalize to [0, 1]
    # Convert from (H, W, C) to (C, H, W) and scale to [0, 1]
    img_tensor_normalized = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor_normalized.to(DEFAULT_DEVICE)

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_tensor_normalized - mean) / std
    img_normalized.to(DEFAULT_DEVICE)

    return img_normalized, img_tensor, 1/scale

def print_frames_analyzed(i, modu):
    if i % modu == 0:
        print(f"{i} frames processed")

# TODO Implement
def prediction_in_box(query_point, box) -> list:
    # Check that the query point is in the box
    if box.shape[0] > 1:
        box = box[0]
    elif box.shape[0] == 0:
        return False
    box = box.flatten()

    if ((box[0] < query_point[0] < box[0] + box[2]) and
        (box[1] < query_point[1] < box[1] + box[3])):
        return True
    else:
        return False

# TODO Send sensor_coord to motor control
def send_coord(sensor_coord) -> None:
    if sensor_coord is None:
        return

    # Normalize the sensor coordinate value
    print(sensor_coord)
    trackCoords(_servoX, _servoY, sensor_coord[0], sensor_coord[1])

def scale_coord(coord, factor):
    return [int(factor*coord[0]), int(factor*coord[1])]

class IOOptions:
    def __init__(self,
                 test_name: str,
                 send_to_board: bool,
                 print_coord: bool,
                 write_out: bool,
                 disp_out: bool,
                 benchmarking: bool,
                 capture: cv2.VideoCapture):
        self.send_to_board = send_to_board
        self.print_coord = print_coord
        self.write_out = write_out
        self.disp_out = disp_out
        self.benchmarking = benchmarking

        if write_out:
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.visualizer = VideoDisplayer("media/t" + test_name, fps, width, height)

        if self.disp_out:
            self.viewer = LiveVideoViewer()

        # State tracking variables. If false start timer, if true, end timer
        if self.benchmarking:
            self.dino_bench_state = False
            self.dino_timer = None
            self.cotracker_bench_state = False
            self.cotracker_timer = None

    def __del__(self):
        if self.write_out:
            self.visualizer.save_video()

    def bench_DINO(self) -> None:
        if not self.benchmarking:
            return

    def bench_CoTracker(self) -> None:
        if not self.benchmarking:
            return

    def run(self, sensor_coord, normalized_coord, frame=None) -> None:
        if self.send_to_board:
            send_coord(normalized_coord)

        if self.write_out:
            self.visualizer.add_frame(sensor_coord, frame)

        if self.disp_out:
            self.viewer.show_frame(sensor_coord, frame)

        if self.print_coord:
            if sensor_coord is None:
                print("No Point")
            else:
                print(f"x: {sensor_coord[0]:.2f} | y: {sensor_coord[1]:.2f}")

class LiveVideoViewer:
    def __init__(self):
        cv2.namedWindow('Live View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live View', 800, 600)

    def show_frame(self, pred, frame):
        if pred is not None:
            cv2.circle(
                frame,
                (int(pred[0]), int(pred[1])),
                radius=5,
                color=(0, 255, 0),
                thickness=-1,
            )

        cv2.imshow('Live View', frame)

        # Wait for 25ms and check if 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return

class VideoDisplayer:
    def __init__(self, filename, fps, width, height):
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(filename+".mp4", self.fourcc, self.fps, (self.width, self.height))

    def add_frame(self, pred, frame):
        if pred is not None:
            cv2.circle(
                frame,
                (int(pred[0]), int(pred[1])),
                radius=5,
                color=(0, 255, 0),
                thickness=-1,
            )

        self.writer.write(frame)

    def save_video(self):
        self.writer.release()
