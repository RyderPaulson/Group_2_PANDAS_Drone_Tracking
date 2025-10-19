# PyPi packages
import torch
import numpy as np

# From cloned repositories
from groundingdino.util.inference import load_model, predict

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

class GroundingDINOCORE:
    """
    Class for detecting a drone in an image using GroundingDINO.
    """
    def __init__(self, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        Initialize GroundingDINO.
        :param text_prompt:
        :param box_threshold:
        :param text_threshold:
        """
        self.model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                "GroundingDINO/weights/groundingdino_swint_ogc.pth")
        self.model.to(DEFAULT_DEVICE)
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, frame: torch.tensor):
        boxes, logits, phrases = predict(
            model=self.model,
            image=frame,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        return boxes, logits, phrases

def find_sensor(img, box, target_color=[128, 128, 128], th=0.3):
    """
    Given a bounding box around the drone as obtained by GroundingDINO, will find the specific point to feed into
    CoTracker.
    Note, since color is an important distinguisher, we do not convert the image to gray.
    :param img: image to annotate. Must be in RGB format
    :param box: can either be in tensor form or already converted to a 1x4 numpy array.
    :param shape: the shape of the source image
    :return query_point: [x, y] coordinates
    """
    # If more than one box is found, reduce to only the highest weighted one
    if box.shape[0] > 1:
        box = box[0]
    elif box.shape[0] == 0:
        return np.array([0, 0])
    box = box.flatten()

    # Get image dimensions as tensors
    img_h, img_w = img.shape[1], img.shape[2]

    # Convert normalized box coordinates to pixel coordinates
    x_min = (box[0] * img_w - 0.5 * box[2] * img_w).int()
    y_min = (box[1] * img_h - 0.5 * box[3] * img_h).int()
    x_max = (box[0] * img_w + 0.5 * box[2] * img_w).int()
    y_max = (box[1] * img_h + 0.5 * box[3] * img_h).int()

    # Clamp coordinates to image bounds to avoid possible rounding error
    # This is important because CoTracker can actually track off of the
    # screen so we still want to be able to handle that
    x_min = torch.clamp(x_min, min=0, max=img_w)
    y_min = torch.clamp(y_min, min=0, max=img_h)
    x_max = torch.clamp(x_max, min=0, max=img_w)
    y_max = torch.clamp(y_max, min=0, max=img_h)

    # Crop the image to the bounding box
    cropped_img = img[:, y_min:y_max, x_min:x_max].to(DEFAULT_DEVICE)

    # Define threshold values as tensor
    target_color_tensor = torch.tensor(target_color, dtype=torch.uint8, device=DEFAULT_DEVICE)
    low_th = (target_color_tensor * (1 - th)).view(3, 1, 1).to(DEFAULT_DEVICE)
    high_th = (target_color_tensor * (1 + th)).view(3, 1, 1).to(DEFAULT_DEVICE)

    # Create color mask within the cropped region
    th_mask = torch.all((cropped_img >= low_th) & (cropped_img <= high_th), dim=0)

    # Get coordinates of masked pixels (in cropped space)
    coordinates = torch.argwhere(th_mask)  # Returns (y, x) format

    if len(coordinates) == 0:
        # No pixels found, return center of bounding box
        center_x = (box[0] * img_w).int().item()
        center_y = (box[1] * img_h).int().item()
        return np.array([center_x, center_y])

    # Average all identified points to find the central point
    mean_coords = torch.mean(coordinates.float(), dim=0)

    # Convert coordinates back to src frame
    query_point_x = (mean_coords[1] + x_min).int().item()
    query_point_y = (mean_coords[0] + y_min).int().item()

    return np.array([query_point_x, query_point_y])
