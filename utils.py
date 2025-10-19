# PyPi packages
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def preprocess_frame(img_array):
    """
    img_array: numpy array (H, W, C) with color values in [0, 255]
    returns: torch tensor (C, H, W) normalized
    """
    # Resize (values are based on groundedDINO load_img function
    h, w = img_array.shape[:2]
    target_size = 800
    max_size = 1333

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
    # Normalize the sensor coordinate value

    return

def scale_coord(coord, factor):
    return [int(factor*coord[0]), int(factor*coord[1])]

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


class PointSelecter:
    """
    Implementation of Bishoy's point selection utility in an object-oriented format.
    """

    def __init__(self):
        # Global variables for point selection
        self.selected_points = []
        self.selection_mode = "points"  # "points", "area", or "quad"
        self.area_corners = (
            []
        )  # For area selection: rectangle [top_left, bottom_right] or quad [p1, p2, p3, p4]
        self.frame_for_selection = None
        self.fig, self.ax = None, None
        self.polygon = None  # For displaying the polygon during area selection
        self.grid_size = 10  # Default grid size, will be updated from args
        self.resize_factor = (
            1.0  # Resize video to this fraction of original size (0.25 = quarter size)
        )
        self.max_frames = 100  # Maximum number of frames to process at once

    def sort_corners(self, corners):
        """Sort the corners in a consistent order: top-left, top-right, bottom-right, bottom-left.

        Args:
            corners: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            Sorted corners in the order [top-left, top-right, bottom-right, bottom-left]
        """
        # Convert to numpy array for easier manipulation
        corners_array = np.array(corners)

        # Find the center of the quadrilateral
        center = np.mean(corners_array, axis=0)

        # Calculate the angles from center to each corner
        angles = np.arctan2(
            corners_array[:, 1] - center[1], corners_array[:, 0] - center[0]
        )

        # Sort corners by angle (counterclockwise from the right)
        sorted_indices = np.argsort(angles)
        sorted_corners = [corners[i] for i in sorted_indices]

        # Rotate so that the top-left corner is first
        # Find the corner with the smallest y value (top)
        top_corners = sorted(
            sorted_corners[:2], key=lambda p: p[0]
        )  # Sort left to right
        bottom_corners = sorted(
            sorted_corners[2:], key=lambda p: p[0], reverse=True
        )  # Sort right to left

        # Arrange in the order: top-left, top-right, bottom-right, bottom-left
        return [top_corners[0], top_corners[1], bottom_corners[0], bottom_corners[1]]

    def create_grid_in_quad(self, grid_size=10):
        """Create a grid of points within a quadrilateral defined by 4 corner points.

        Args:
            grid_size: Size of the grid (grid_size x grid_size)
        """

        if len(self.area_corners) != 4:
            return

        # Clear any previously selected points
        selected_points = []

        # The corners should already be sorted in the order:
        # p1 = top-left, p2 = top-right, p3 = bottom-right, p4 = bottom-left
        p1 = np.array(self.area_corners[0])  # Top-left
        p2 = np.array(self.area_corners[1])  # Top-right
        p3 = np.array(self.area_corners[2])  # Bottom-right
        p4 = np.array(self.area_corners[3])  # Bottom-left

        # Create a grid of points using bilinear interpolation
        for i in range(grid_size):
            for j in range(grid_size):
                # Normalized coordinates (0 to 1)
                u = i / (grid_size - 1) if grid_size > 1 else 0
                v = j / (grid_size - 1) if grid_size > 1 else 0

                # Bilinear interpolation formula for a quadrilateral
                point = (
                    (1 - u) * (1 - v) * p1
                    + u * (1 - v) * p2
                    + u * v * p3
                    + (1 - u) * v * p4
                )

                # Convert to integer coordinates
                x, y = int(point[0]), int(point[1])
                selected_points.append([x, y])

                # Draw a circle at the grid point
                circle = plt.Circle((x, y), 3, color="green")
                self.ax.add_patch(circle)

        print(
            f"Created {grid_size}x{grid_size} grid with {len(selected_points)} points in quadrilateral"
        )

    def click_event(self, event):
        """Handle mouse click events to select points or area."""

        # Only process actual mouse clicks
        if not hasattr(event, "button") or event.button != 1:
            return  # Not a left-click event

        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside the image

        # Ensure we have valid coordinates
        x, y = int(event.xdata), int(event.ydata)
        if x <= 0 and y <= 0:
            return  # Ignore suspicious clicks at origin

        if event.button == 1:  # Left mouse button
            if self.selection_mode == "points":
                self.selected_points.append([x, y])
                # Draw a circle at the selected point
                circle = plt.Circle((x, y), 5, color="red")
                self.ax.add_patch(circle)
                # Add text label with point index
                self.ax.text(
                    x + 10,
                    y + 10,
                    str(len(self.selected_points)),
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="red", alpha=0.7),
                )
                plt.draw()
                print(f"Selected point {len(self.selected_points)}: ({x}, {y})")

            elif self.selection_mode == "area":
                # Add the corner point
                self.area_corners.append([x, y])
                corner_num = len(self.area_corners)
                print(f"Selected corner {corner_num}: ({x}, {y})")

                # Draw a point for the corner
                circle = plt.Circle((x, y), 5, color="blue")
                self.ax.add_patch(circle)

                # Add text label with corner index
                self.ax.text(
                    x + 10,
                    y + 10,
                    str(corner_num),
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="blue", alpha=0.7),
                )

                # If we have 4 corners, sort them, draw the quadrilateral and create the grid
                if corner_num == 4:
                    # Sort the corners in a consistent order (top-left, top-right, bottom-right, bottom-left)
                    sorted_corners = self.sort_corners(self.area_corners)
                    area_corners = sorted_corners

                    # Draw the sorted quadrilateral
                    xs = [p[0] for p in area_corners]
                    ys = [p[1] for p in area_corners]
                    # Close the polygon by repeating the first point
                    xs.append(xs[0])
                    ys.append(ys[0])

                    # Create polygon
                    polygon = plt.Polygon(
                        area_corners, fill=False, edgecolor="blue", linewidth=2
                    )
                    self.ax.add_patch(polygon)

                    # Redraw the corner labels with the sorted order
                    for i, (x, y) in enumerate(area_corners):
                        # Clear previous text if any
                        for txt in self.ax.texts:
                            if txt.get_position() == (x + 10, y + 10):
                                txt.remove()

                        # Add new text label with sorted corner index
                        self.ax.text(
                            x + 10,
                            y + 10,
                            str(i + 1),
                            color="white",
                            fontsize=12,
                            bbox=dict(facecolor="blue", alpha=0.7),
                        )

                    # Create a grid of points within the quadrilateral
                    self.create_grid_in_quad(self.grid_size)

                plt.draw()

    def select_points(self, frame, grid_size_arg=10):
        """Display a frame and allow user to select points or area.

        Args:
            frame: Video frame to display
            grid_size_arg: Size of the grid for area selection mode
        """

        # Update the global grid_size
        self.grid_size = grid_size_arg

        # Reset selection data
        self.selected_points = []
        self.area_corners = []

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.frame_for_selection = frame

        # Display the frame
        self.ax.imshow(frame)

        # Create radio buttons for selection mode
        from matplotlib.widgets import RadioButtons

        rax = plt.axes([0.05, 0.05, 0.15, 0.10])
        radio = RadioButtons(rax, ("Points", "Area"))

        def mode_change(label):
            if label == "Points":
                self.selection_mode = "points"
                # Clear area selection if any
                self.area_corners = []
                # Remove any polygon if it exists
                for patch in self.ax.patches:
                    if isinstance(patch, plt.Polygon):
                        patch.remove()
            else:  # 'Area'
                self.selection_mode = "area"
                # Clear point selection if any
                for i in range(len(self.ax.patches)):
                    if i >= len(self.ax.patches):
                        break
                    if (
                        isinstance(self.ax.patches[i], plt.Circle)
                        and self.ax.patches[i].get_facecolor()[0] == 1
                    ):  # Red circles
                        self.ax.patches[i].remove()
                # Clear any existing area corners and selected points
                self.area_corners = []
                self.selected_points = []
            plt.draw()

        radio.on_clicked(mode_change)

        if self.selection_mode == "points":
            self.ax.set_title(
                "Click to select individual points for tracking (close window when done)"
            )
        else:
            self.ax.set_title(
                "Click to select four corners of an area for grid (close window when done)"
            )

        # Connect the click event
        self.fig.canvas.mpl_connect("button_press_event", self.click_event)

        # Show the plot (this will block until the window is closed)
        plt.tight_layout()
        plt.show()

        return np.array(self.selected_points)
