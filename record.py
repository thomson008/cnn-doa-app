import pyaudio
import sys
import numpy as np
from scipy.io import wavfile
from utils import CHUNK, RATE, CHANNELS, FORMAT

LEN = int(sys.argv[1])
recording_angle = sys.argv[2]

print(f'Recording data for angle: {recording_angle} degrees.')
print(f'Recording length will be {LEN} seconds.')

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []

for i in range(int(LEN * RATE / CHUNK)):
    data = stream.read(CHUNK)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

print('Saving data...')

frames = np.frombuffer(b''.join(frames), dtype=np.int16)
frames = np.reshape(frames, (-1, CHANNELS))
all_data = frames[:, :CHANNELS - 1]

wavfile.write(f'../training_data/recording_angle_{recording_angle}.wav', RATE, all_data)

print('Done.')
