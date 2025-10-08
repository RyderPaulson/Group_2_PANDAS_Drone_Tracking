import Jetson.GPIO as GPIO
import time
# import argparse
import sys
import select
def main(pin, duty_cycle, freq):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, freq)
    pwm.start(duty_cycle)
    print("Starting PWM")
    try:
        while True:
            if sys.stdin in select.slect([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip()
                if line.lower() == "q":
                    break
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pwm.stop()
        GPIO.cleanup()
        print("Stopped")
# Run on pin 32 at 50% duty cycle at 1 kHz
main(32, 50, 1000)