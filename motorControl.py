import Jetson.GPIO as GPIO
import time
import numpy as np

# constants
FREQUENCY = 50 #Hz
PERIOD_MS = 1000 / FREQUENCY
MIN_PULSE_WIDTH_MS = 0.5 # in % duty cycle for 0°
MAX_PULSE_WIDTH_MS = 2  # in % duty cycle for 180°

# --- Low Level Math Functions (Conversions/Formulas) ---

# Converts angle (0-180) to pulse (MIN_PULSE_WIDTH_MS to MAX_PULSE_WIDTH_MS)
def angle_to_pulse(angle):
    angle = max(0, min(180, angle))    # ensure within 90-140 deg
    pulse_ms = MIN_PULSE_WIDTH_MS + (angle / 180.0) * (MAX_PULSE_WIDTH_MS - MIN_PULSE_WIDTH_MS)
    return pulse_ms

# Converts pulse to duty cycle
#       Note: pulse width is the actual duration of the "high signal" and
#             duty cycle is the percentage of the total PWM period that the signal is high (used by Jetson.GPIO library)
def pulse_to_duty(pulse_ms):
    return (pulse_ms / PERIOD_MS) * 100


# --- High Level Servo Functions (Called by CoTracker or manually) ---
class servo:
    def __init__(self, pin):
        self.pin = pin
        # self.dx = 0
        self.dx =90
        self.sum = 0

    def getPin(self):
        return self.pin

    # alpha: how much to over/undershoot
    # beta: scale factor of rotations 
    # def servoMoveExp(self, coord, alpha=0.5, beta=1):
      
    #     angle = coord * 50 * + 90
    #     self.dx = coord
    #     # angle = self.dx * 90 + 90
    #     # self.sum = alpha * (self.sum + angle)
    #     # output_angle = beta * self.sum
    #     # self.dx = coord

    #     print(f"angle={angle:.2f}°")

    #     return angle
    
    def servoMove(self, coord, kd=-0.5):
        # Add between -15*correction -> 15*correction deg to previous
        angle = 15*coord + 15 * kd*(coord-self.dx) + self.dx
        angle = max(0, min(180, angle)) 
        self.dx = angle
        # angle = self.dx * 90 + 90
        # self.sum = alpha * (self.sum + angle)
        # output_angle = beta * self.sum
        # self.dx = coord

        print(f"angle={angle:.2f}°")

        return angle
    

    # Rotate the motor to a specific angle. (immediate, no decay)
    def setServo(self, angle):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.HIGH)
        pwm = GPIO.PWM(self.pin, FREQUENCY)
        
        duty_cycle = pulse_to_duty(angle_to_pulse(angle))

        pwm.start(duty_cycle)
        time.sleep(1)
        pwm.stop()
        GPIO.cleanup()
        print("Set servo to ", angle, " deg.")

        # Rotate the motor to a specific angle. (immediate, no decay)
    def setServoGivenPulseMS(self, pulsems):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.HIGH)
        pwm = GPIO.PWM(self.pin, FREQUENCY)
        
        duty_cycle = pulse_to_duty(pulsems/1000)
        pwm.start(3)
        time.sleep(3)
        pwm.stop()
        GPIO.cleanup()
        print("Set servo to ", 45, " deg.")


def trackCoords(servoX, servoY, dx, dy):
    pin1 = servoX.getPin()
    pin2 = servoY.getPin()
    s1Angle = servoX.servoMove(dx)
    s2Angle = servoY.servoMove(dy)

    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(pin1, GPIO.OUT, initial=GPIO.HIGH)
    pwm1 = GPIO.PWM(pin1, FREQUENCY)
    pwm1.start(pulse_to_duty(angle_to_pulse(s1Angle)))

    GPIO.setup(pin2, GPIO.OUT, initial=GPIO.HIGH)
    pwm2 = GPIO.PWM(pin2, FREQUENCY)
    pwm2.start(pulse_to_duty(angle_to_pulse(s2Angle)))

    time.sleep(.08)
    
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()


# import cv2
# import threading

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # Get camera properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define codec and create VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# # Flag to control recording
# recording = True

# def record_video():
#     """Function to continuously record frames"""
#     while recording:
#         ret, frame = cap.read()
#         if ret:
#             out.write(frame)
#         else:
#             break

# # Start recording in a separate thread
# record_thread = threading.Thread(target=record_video)
# record_thread.start()

# print("Recording started...")

# ========================================


# cameraServo = servo(32)
# baseServo = servo(33)

# cameraServo.setServo(180)
# baseServo.setServo(0)

# [trackCoords(cameraServo, baseServo, i, 0) for i in np.arange(0, 1, 0.01)]

# [trackCoords(cameraServo, baseServo, 0, 1) for _ in np.arange(1, 100, 1)]
# [trackCoords(cameraServo, baseServo, -1, 0) for _ in np.arange(1, 100, 1)]
# [trackCoords(cameraServo, baseServo, 0, -1) for _ in np.arange(1, 100, 1)]


# [trackCoords(cameraServo, baseServo, 0, 0 ) for i in np.arange(0, 0.4, 0.01)]
# input("")


# input("")
# cameraServo.setServo(60)
# cameraServo.setServo(40)

# ========================================

# # Stop recording
# recording = False
# record_thread.join()  # Wait for the recording thread to finish

# # Release resources
# cap.release()
# out.release()

# print("Recording stopped and saved")




# cameraServo.setServo(170)
# cameraServo.setServo(160)
# cameraServo.setServo(150)
# cameraServo.setServo(140)
# cameraServo.setServo(130)
# cameraServo.setServo(120)
# baseServo.setServo(120)


# cameraServo.setServoGivenPulseMS(400)
# baseServo.setServoGivenPulseMS(400)
# baseServo.setServo(0)

#for i in range(100):
#    trackCoords(baseServo, cameraServo, 1, 1)


# servo.servoMoveExp(32, 1)
# servo.servoMoveExp(32, .8)
# servo.servoMoveExp(32, .2)
# servo.servoMoveExp(32, .001)
# servo.servoMoveExp(32, -1)
# servo.servoMoveExp(32, -.8)
# servo.servoMoveExp(32, -.2)
# servo.servoMoveExp(32, 0)
# servo.servoMoveExp(32, 0)
