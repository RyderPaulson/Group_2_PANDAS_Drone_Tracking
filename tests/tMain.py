from main import main
import unittest

class TestMain(unittest.TestCase):
    def test_ds_pan_cut(self):
        test_name = "main_ds_pan_cut"
        src_path = "media/ds_pan_cut.mp4"
        main(src_path,
             test_name=test_name,
             print_coord=True,
             send_to_board=True,
             disp_out=True,
             )

    def test_ds_frame(self):
        test_name = "main_ds_frame"
        src_path = "media/ds_frame.mp4"
        main(src_path,
             test_name=test_name,
             disp_out=True,
        )

    def test_ds_pan_full(self):
        test_name = "main_ds_pan"
        src_path = "media/ds_pan.mp4"
        main(src_path,
             test_name=test_name,
             disp_out=True
        )

    def test_live_video(self):
        test_name = "main_live"
        src_path = 0
        main(
            src_path,
            text_prompt="human head", 
            test_name=test_name,
            disp_out=True
        )

    def test_motor_control(self):
        test_name = "main_motor_control"
        src_path = "media/ds_pan_cut.mp4"
        main(src_path,
             test_name=test_name,
             print_coord=True,
             send_to_board=True,
        )

    def test_ozz(self):
        test_name = "oz"
        src_path = "media/SuccessfulOz.mp4"
        main(src_path,
             test_name=test_name,
             disp_out=True,
             write_out=True,
             target_color=[54, 36, 61]
        )
