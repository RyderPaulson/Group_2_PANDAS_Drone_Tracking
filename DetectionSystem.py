# PyPi packages
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

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
        :param text_prompt: Text prompt for detection (e.g., "a drone")
        :param box_threshold: Confidence threshold for bounding boxes (used as threshold parameter)
        :param text_threshold: Not used in HuggingFace implementation
        """
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            DEFAULT_DEVICE
        )

        # IMPORTANT: text prompts need to be lowercased and end with a dot
        self.text_prompt = text_prompt.lower()
        if not self.text_prompt.endswith("."):
            self.text_prompt += "."

        # HuggingFace API only uses a single threshold parameter
        self.threshold = box_threshold

    def detect(self, frame: torch.Tensor):
        """
        Detect objects in the frame and return only the highest-scoring detection.
        :param frame: Input image as torch tensor (C, H, W) in RGB format with values 0-255
        :return: boxes, logits, phrases (single detection with highest score, or empty if no detections)
        """
        # Convert tensor to PIL Image
        if frame.dtype != torch.uint8:
            frame = frame.byte()

        # Convert from (C, H, W) to (H, W, C) and then to PIL
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(frame_np)

        # Get image size for post-processing
        target_size = [image.size[::-1]]  # (height, width)

        # Process inputs
        inputs = self.processor(
            images=image, text=self.text_prompt, return_tensors="pt"
        ).to(DEFAULT_DEVICE)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results - FIXED: Use correct API with threshold parameter
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_size
        )[
            0
        ]  # Get first (and only) image results

        # Extract boxes, scores, and labels
        boxes = results["boxes"]  # In xyxy format (pixel coordinates)
        logits = results["scores"]
        phrases = results["labels"]

        # Handle case when no detections are found
        if len(boxes) == 0:
            return torch.empty((0, 4)), torch.empty(0), []

        # Find the index of the highest scoring detection
        max_score_idx = torch.argmax(logits)

        # Keep only the highest scoring detection
        boxes = boxes[max_score_idx:max_score_idx+1]
        logits = logits[max_score_idx:max_score_idx+1]
        phrases = [phrases[max_score_idx]]

        # Convert boxes from xyxy (pixel coords) to cxcywh (normalized) format to match original API
        h, w = target_size[0]

        boxes_cxcywh = torch.zeros_like(boxes)
        boxes_cxcywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2 / w  # center_x
        boxes_cxcywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 / h  # center_y
        boxes_cxcywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w  # width
        boxes_cxcywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h  # height

        return boxes_cxcywh, logits, phrases


def find_sensor(img, box, target_color=[128, 128, 128], th=0.3):
    """
    Given a bounding box around the drone as obtained by GroundingDINO, will find the specific point to feed into
    CoTracker.
    Note, since color is an important distinguisher, we do not convert the image to gray.
    :param img: image to annotate. Must be in RGB format
    :param box: can either be in tensor form or already converted to a 1x4 numpy array.
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
    x_min = torch.clamp(x_min, min=0, max=img_w)
    y_min = torch.clamp(y_min, min=0, max=img_h)
    x_max = torch.clamp(x_max, min=0, max=img_w)
    y_max = torch.clamp(y_max, min=0, max=img_h)

    # Crop the image to the bounding box
    cropped_img = img[:, y_min:y_max, x_min:x_max].to(DEFAULT_DEVICE)

    # Define threshold values as tensor
    target_color_tensor = torch.tensor(
        target_color, dtype=torch.uint8, device=DEFAULT_DEVICE
    )
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