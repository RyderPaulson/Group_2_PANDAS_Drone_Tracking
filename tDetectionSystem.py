# PyPi packages
import unittest
import cv2
import torch
import numpy as np

# From cloned repositories
from groundingdino.util.inference import load_image, annotate

# Unique to project
from DetectionSystem import GroundedDINOCORE, find_sensor
from utils import normalize_np_img

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

class TestDetectionSystem(unittest.TestCase):
    def test_drone_detection_frame(self):
        text_prompt = 'blue drone'

        img_src, img  = load_image("media/ds_pan_f1.png")
        grounding_dino = GroundedDINOCORE(text_prompt)
        boxes, logits, phrases = grounding_dino.detect(img, [0, 0])
        img_annotated = annotate(image_source=img_src, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite("media/ds_pan_f1_a.png", img_annotated)

    def test_drone_detection_video(self):
        # Test parameters
        text_prompt = 'drone'
        vid_path = 'media/ds_pan_cut.mp4'
        vid_out_path = 'media/d_detect_v1.mp4'

        # Create input capture obj
        cap = cv2.VideoCapture(vid_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get characteristics of input video and start output obj
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(vid_out_path, fourcc, fps, (width, height))

        # Start GroundingDINO
        grounding_dino = GroundedDINOCORE(text_prompt)

        for i in range(video_length-1):
            # Pull frame from capture
            ret, frame_src  = cap.read()

            # Break if the capture object was not able to get a frame
            if not ret:
                break

            # Normalize image for the sake of groundingDINO
            frame = normalize_np_img(frame_src)

            # Detect boxes with groundingDINO
            boxes, logits, phrases = grounding_dino.detect(frame, None)

            # Annotate frame with GroundingDINO library
            frame_annotated = annotate(image_source=frame_src, boxes=boxes, logits=logits, phrases=phrases)

            # Write to output
            out.write(frame_annotated)

        # Release all capture and reading objects
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def test_sensor_identification(self):
        # Standard workflow for getting a bounding box prediction from GroundingDINO
        text_prompt = "blue drone"
        img_path = "media/ds_pan_f1.png"
        output_image, img = load_image(img_path)
        grounding_dino = GroundedDINOCORE(text_prompt)
        boxes, logits, phrases = grounding_dino.detect(img, [0, 0])
        img_annotated = annotate(
            image_source=output_image, boxes=boxes, logits=logits, phrases=phrases
        )

        # Target color determined from looking at the image manually
        target_color = [87, 41, 62]

        # Find query point and print it
        query_point = find_sensor(output_image, boxes, target_color=target_color)
        print(query_point)

        output_image = output_image.copy()

        # Draw a circle at the query point
        cv2.circle(
            output_image,
            (int(query_point[0]), int(query_point[1])),
            radius=5,
            color=(0, 255, 0),
            thickness=-1,
        )

        # Save the final annotated image
        cv2.imwrite("media/ds_pan_f1_qp.png", output_image)


if __name__ == "__main__":
    unittest.main()
