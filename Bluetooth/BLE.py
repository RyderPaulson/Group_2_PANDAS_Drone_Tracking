#!/usr/bin/env/python3

import asyncio
from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
import socket

PHOTOMETER = "78:21:84:7A:5D:0A"
UUID = "184E"

IP = "127.0.0.1"
PORT = 10001

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def notification_handler(characteristic: BleakGATTCharacteristic, packet: bytearray):
    if (L := len(packet)) != 5:
        print(f"MALFORMED PACKET (LENGTH {L}): CONTINUING")
    
    timestamp = int.from_bytes(packet[0:4], byteorder='big', signed=False)
    data = int(packet[4])

    print(f"{timestamp:0{32}d} : {data}")
    s.sendto(packet, (IP, PORT))

async def main(address: str, characteristic: str):
    while True:
        try:
            async with BleakClient(address) as client:
                try:
                    await client.start_notify(characteristic, notification_handler)
                    print(f"CONNECTION ESTABLISHED: CALLBACK INITIATED")
                    while True:
                        await asyncio.sleep(1)
                except:
                    await client.stop_notify(characteristic)
                print(f"BLUETOOTH DISCONNECTED: RETRYING")
        except TimeoutError:
            print(f"FAILURE TO CONNECT: RETRYING")
        except OSError:
            print(f"BLUETOOTH NOT SUPPORTED: TERMINATING")
            break
        except:
            print(f"UNKNOWN ERROR: TERMINATING")
            break

asyncio.run(main(PHOTOMETER, UUID))
