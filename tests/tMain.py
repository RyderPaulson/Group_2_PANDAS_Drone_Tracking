from main import main
import unittest

class TestMain(unittest.TestCase):
    def test_ds_pan_cut(self):
        test_name = "main_ds_pan_cut"
        src_path = "../media/ds_pan_cut.mp4"
        main(src_path,
             test_name=test_name,
             write_out=True,
             print_coord=True,
             benchmarking=True,
             )

    def test_ds_frame(self):
        test_name = "main_ds_frame"
        src_path = "../media/ds_frame.mp4"
        main(src_path,
             test_name=test_name,
             disp_out=True,
             benchmarking=True,
             )

    def test_ds_pan_full(self):
        test_name = "main_ds_pan"
        src_path = "../media/ds_pan.mp4"
        main(src_path,
             test_name=test_name,
             print_coord=True
             )

    def test_line_video(self):
        test_name = "main_ds_pan"
        src_path = 0
        main(
            src_path,
            text_prompt="human head", 
            test_name=test_name,
            disp_out=True
        )
