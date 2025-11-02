import Jetson.GPIO as GPIO
import time
import numpy as np

# constants
FREQUENCY = 50 #Hz
PERIOD_MS = 1000 / FREQUENCY
MIN_PULSE_WIDTH_MS = 0.05 # in % duty cycle for 0°
MAX_PULSE_WIDTH_MS = 1.6  # in % duty cycle for 180°
# NOTE: The vertical range of the turret is 60 to 180 degrees.

current_angle = 60 # start at neutral


# --- Low Level Math Functions (Conversions/Formulas) ---

# Converts angle (0-180) to pulse (MIN_PULSE_WIDTH_MS to MAX_PULSE_WIDTH_MS)
def angle_to_pulse(angle):
    angle = max(60, min(180, angle))    # ensure within 60-180 deg
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
        self.dx = 0
        self.sum = 0
        
    def servoMoveExp(self, pin, coord, alpha=0.5):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
        pwm = GPIO.PWM(pin, FREQUENCY)
        pwm.start(pulse_to_duty(angle_to_pulse(current_angle)))
        try:
            angle = self.dx * 90 + 90
            self.sum = alpha*(self.sum + angle)
            beta = 1
            output_angle = beta * self.sum
            output_angle = max(60, min(180, output_angle))    #ensure within 60-180 deg
            self.dx = coord
            pwm.ChangeDutyCycle(pulse_to_duty(angle_to_pulse(output_angle)))
    
            # update angle
            print(f"angle={output_angle:.2f}°")
            time.sleep(0.1)
        finally:
        pwm.stop()
        GPIO.cleanup()
    # Rotate the motor to a specific angle. (immediate, no decay)
    def setServo(self, pin, angle):
        global current_angle
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
        pwm = GPIO.PWM(pin, FREQUENCY)
        
        duty_cycle = pulse_to_duty(angle_to_pulse(angle))
        pwm.start(duty_cycle)
        time.sleep(1)
        pwm.stop()
        GPIO.cleanup()
        current_angle = angle
        print("Set servo to ", angle, " deg.")

# Applies exp decay function to a coordinate buffer
# NOTE: Increase alpha if you want the servo to accelerate quickly toward the target. Decrease alpha if you want a more gradual transition.
# def servoMoveExp(pin, dx, alpha=0.5):
#     global current_angle

#     GPIO.setmode(GPIO.BOARD)
#     GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
#     pwm = GPIO.PWM(pin, FREQUENCY)
#     pwm.start(pulse_to_duty(angle_to_pulse(current_angle)))
    
#     try:
#         target_angle = current_angle + dx
#         print("Target angle: ", target_angle, "°")

#         for step in range(steps):
#             # This uses the exponetial decay formula: 1 - e^(-kt)
#             #   Here, -k is alpha and t is the time, or here, proportion of step until finished
#             fraction = (1 - np.exp(-alpha * ((step + 1) / steps)))  # amount to change per step
            
#             current_angle += (target_angle - current_angle) * fraction

#             current_angle = max(60, min(180, current_angle))    #ensure within 60-180 deg
            
#             # ypdate angle
#             pwm.ChangeDutyCycle(pulse_to_duty(angle_to_pulse(current_angle)))

#             print(f"Step {step+1}/{steps} → angle={current_angle:.2f}°")
#             time.sleep(0.1)

#         pwm.ChangeDutyCycle(pulse_to_duty(angle_to_pulse(target_angle))) #ensures exact angle always reached

#     finally:
#         pwm.stop()
#         GPIO.cleanup()

# setServo(32,60)     # set angle to 60 degrees
# servoMoveExp(32, 120)   # the camera should now be 180 degrees (60+120)
# setServo(32,180)

servo = servo(32)
servo.servoMoveExp(32, 1)
servo.servoMoveExp(32, .8)
servo.servoMoveExp(32, .2)
servo.servoMoveExp(32, .001)
servo.servoMoveExp(32, -1)
servo.servoMoveExp(32, -.8)
servo.servoMoveExp(32, -.2)
servo.servoMoveExp(32, 0)
servo.servoMoveExp(32, 0)
