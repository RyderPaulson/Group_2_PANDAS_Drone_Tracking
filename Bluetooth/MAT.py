#!/usr/bin/env/python3

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import socket

UPDATE_INTERVAL = 40
DISPLAY_DEPTH = 100

FIG = plt.figure()
A = FIG.add_subplot(1, 1, 1)
X = []
Y = []

IP = "127.0.0.1"
PORT = 10001

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((IP, PORT))

def update_plot(i, x: list[int], y: list[int]):
    try:
        packets, addr = s.recvfrom(5000)
        
        if not packets:
            return
        
        for i in range(len(packets)//5):
            packet = packets[5*i:5*(i+1)]

            timestamp = int.from_bytes(packet[0:4], byteorder='big', signed=False)
            reading = int(packet[4])

            data = (timestamp, reading)

            x.append(data[0]) # type: ignore
            y.append(data[1]) # type: ignore

            x = x[-DISPLAY_DEPTH:]
            y = y[-DISPLAY_DEPTH:]

    except:
        print("UNKNOWN DISPLAY ERROR: CONTINUING")

    A.clear()
    A.plot(x, y)

    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Received Voltage')
    plt.ylabel('Relative Magnitude')
    plt.ylim((0,255))

ani = FuncAnimation(FIG, update_plot, fargs=(X, Y), interval=UPDATE_INTERVAL) # type: ignore
plt.show()
