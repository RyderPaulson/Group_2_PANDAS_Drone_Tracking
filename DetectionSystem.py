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

class GroundedDINOCORE:
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

    def detect(self, frame, previous_qp):
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
    :param img: -> image to annotate. Must be in RGB format
    :param box: -> can either be in tensor form or already converted to a 1x4 numpy array.
    :return query_point: -> [x, y] coordinates
    """
    if type(box) == torch.Tensor:
        box = box.numpy()[0]

    # Give that the box dimensions are given as a normalized value from 0-1 find the actual pixel values of the box
    img_shape = img.shape
    img_h = img_shape[0]
    img_w = img_shape[1]
    box = np.array([img_w * box[0] - 0.5*(img_w * box[2]),
                    img_h * box[1] - 0.5*(img_h * box[3]),
                    img_w * box[0] + 0.5*(img_w * box[2]),
                    img_h * box[1] + 0.5*(img_h * box[3])])
    box = box.astype(int)

    # Create a max of just the bounding box
    rows, cols = np.ogrid[:img.shape[0], :img.shape[1]]
    box_mask = (
        (rows >= box[1]) & (rows < box[3]) & (cols >= box[0]) & (cols < box[2])
    )

    # Define threshold values
    low_th = [int(i - i * th) for i in target_color]
    high_th = [int(i + i * th) for i in target_color]

    # Create mask with all pixels within the threshold
    th_mask = np.all((low_th <= img) & (img <= high_th), axis=-1)

    # Combine the masks and convert to just be list of coordinates
    mask = box_mask & th_mask
    coordinates = np.argwhere(mask)

    # Average all identified points in the mask to find the query point
    query_point = np.mean(coordinates, axis=0)[::-1]
    query_point = query_point.astype(int)

    return query_point
