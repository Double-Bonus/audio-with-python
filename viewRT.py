import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt


print("Labas, veikia ;)")

# constants
CHUNK = 1024 * 4             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

data = stream.read(CHUNK)  # gives hex data!
# print(data)

data_int = struct.unpack(str(2 * CHUNK) + 'B', data) # gives uint8_t range

print(data_int)

fig, ax = plt.subplots()

ax.plot(data_int)
plt.show()

