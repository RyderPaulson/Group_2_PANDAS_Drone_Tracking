from groundingdino.util.inference import load_model, predict

class DroneDetector:
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

class FindSensor:
    """

    """
    def __init__(self):
        # Initializes sensor locating system
        pass

    def mask(self):
        """
        Creates a mask over the appropriately color tape only looking within the box identified by GroundingDINO.
        :return:
        """
        pass

    def find_point(self):
        """
        Averages all the points in the mask to find the specific point that CoTracker will track.
        :return:
        """
        pass
