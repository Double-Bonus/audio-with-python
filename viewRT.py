import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt


print("Labas, veikia ;)")

# constants
CHUNK = 1024 * 2             # samples per frame
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
    frames_per_buffer=CHUNK,
    input_device_index = 1
)


fig, ax = plt.subplots()

x = np.arange(0, 2 * CHUNK, 2)
# line, = ax.plot(x, np.random.rand(CHUNK)) 
line, = ax.plot(x, np.random.rand(CHUNK), '-', lw=2) # just init
ax.set_ylim(0, 255 * 2)
ax.set_xlim(0, 2 * CHUNK)
plt.setp(ax, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])
plt.show(block=False)

while True:
    data = stream.read(CHUNK)  # gives hex data!
    # print(data)
    data_int = np.array(struct.unpack(str(2 * CHUNK) + 'B', data)) #?? gives uint8_t range
    # print(data_int)
    
    data_np = np.array(data_int, dtype='b')[::2] + 128
    
    line.set_ydata(data_np)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.show()
# 


ax.plot(data_int)
plt.show()

