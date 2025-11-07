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
    angle = max(0, min(180, angle))    # ensure within 0-180 deg
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

    def getPin(self):
         return self.pin
        
    # alpha: how much to over/undershoot
    # beta: scale factor of rotations 
    def servoMoveExp(self, coord, alpha=0.5, beta = 1):
            angle = self.dx * 90 + 90
            self.sum = alpha*(self.sum + angle)
            output_angle = beta * self.sum
            output_angle = max(45, min(120, output_angle))    #ensure within 0-180 deg
            self.dx = coord

            print(f"angle={output_angle:.2f}°")

            return output_angle
    

    # Rotate the motor to a specific angle. (immediate, no decay)
    def setServo(self, angle):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.HIGH)
        pwm = GPIO.PWM(self.pin, FREQUENCY)
        
        duty_cycle = pulse_to_duty(angle_to_pulse(angle))
        pwm.start(duty_cycle)
        time.sleep(3)
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

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin1, GPIO.OUT, initial=GPIO.HIGH)
    pwm1 = GPIO.PWM(pin1, FREQUENCY)
    GPIO.setup(pin2, GPIO.OUT, initial=GPIO.HIGH)
    pwm2 = GPIO.PWM(pin2, FREQUENCY)

    s1Angle = servoX.servoMoveExp(dx)
    s2Angle = servoY.servoMoveExp(dy)

    pwm1.start(pulse_to_duty(angle_to_pulse(s1Angle)))
    pwm2.start(pulse_to_duty(angle_to_pulse(s2Angle)))

    time.sleep(.1)
    
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()

cameraServo = servo(32)
baseServo = servo(33)

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
