import Jetson.GPIO as GPIO
import time
import sys
import select
import numpy as np

# Calculates PWM pulse width from desired angle
def setServoAngle(angle):
    if (angle > 180): 
        angle = 180
    if (angle < 0 ):
        angle = 0
    # 1 ms pulse min for 50 Hz
    global minPulseWidth
    # minPulseWidth = 5
    minPulseWidth=3.8
    # 2 ms pulse max for 50 Hz
    global maxPulseWidth
    maxPulseWidth = 10
    pulse = ((angle * (maxPulseWidth - minPulseWidth)) / 180) + minPulseWidth;
    return pulse

# Applies exponential decay weights to a series of inputs
def expDecay(dx):
    alpha = 0.4
    beta = 180
    pulse = setServoAngle(beta*alpha*dx + 90)
    return pulse

# Rotate the motor to a specific angle.
def setServo(pin, angle):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
    pwm = GPIO.PWM(pin, 50)
    pulse = setServoAngle(angle)
    pwm.start(pulse)
    print(pulse)
    print("Starting PWM")
    # try:
    #     while True:
    #         if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
    #             line = sys.stdin.readline().strip()
    #             if line.lower() == "q":
    #                 break
    #             time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    time.sleep(1)
    pwm.stop()
    GPIO.cleanup()
    print("Stopped")

# Applies decay function to a coordinate buffer
def servoMove(pin, dx):
    pulse = expDecay(dx)
    # Instantiate GPIO pin
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
    pwm = GPIO.PWM(pin, 50)
    # Set angle between 0 - 180 deg according to coordinate buffer
    # print(pulse)
    try:
        pwm.start(pulse)
        angle = (pulse - minPulseWidth)*180/(maxPulseWidth - minPulseWidth)
        print(angle)
        driveFrequency = 20 #Hz
        time.sleep(1/driveFrequency)
        pwm.stop()
    finally:
        pwm.stop()
        GPIO.cleanup()
        # print("Stopped")

# Simulate receiving coordinate buffer from tracking module and move accordingly
# def simulate():
#     servoMove(32, 1)
#     time.sleep(1)
#     dx.append(1)
#     dx.append(.8)
#     dx.append(.5)
#     dx.append(.2)
#     dx.append(.001)
#     servoMove(32, dx)
#     # Reset
#     time.sleep(1)
#     setServo(32, 0)

setServo(32,50)
time.sleep(1)
servoMove(32,1)
# servoMove(32,.8)
# servoMove(32,.2)
# servoMove(32,.2)
# servoMove(32,.2)
# servoMove(32,-1)