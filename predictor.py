import platform
from threading import Thread

from alsa_suppress import noalsaerr
from utils import *


def get_mic_data(in_data):
    data = np.frombuffer(in_data, dtype=np.int16)
    data = np.reshape(data, (-1, CHANNELS))
    # Drop irrelevant channels and reorder remaining channels,
    # in order to match the simulated microphone array
    mic_data = np.hstack([data[:, 1].reshape(-1, 1), data[:, -2:1:-1]])
    return data, mic_data


def get_input_matrix(mic_data):
    gcc_matrix = np.transpose(compute_gcc_matrix(mic_data))
    input_data = np.array([gcc_matrix], dtype=np.float32)
    return input_data


def get_model_details(az_interpreter):
    az_input_details = az_interpreter.get_input_details()[0]
    az_output_details = az_interpreter.get_output_details()[0]
    az_input_shape = az_input_details['shape']
    az_output_shape = az_output_details['shape']
    return az_input_details, az_input_shape, az_output_details, az_output_shape


class Predictor:
    def __init__(self, lines, fig, thresh=50, max_silence_frames=10):
        # Model parameters
        self.is_active = False

        # Thresholds for deciding whether to run or not
        self.thresh = thresh
        self.silent_frames = 0
        self.max_silence_frames = max_silence_frames

        if platform.system() == 'Windows':
            self.p = pyaudio.PyAudio()
        else:
            with noalsaerr():
                self.p = pyaudio.PyAudio()

        self.lines = lines
        self.fig = fig

        self.cnn_exec_times = []
        self.mic_data = np.zeros((CHUNK, CHANNELS - 2))

        if platform.system() == 'Windows':
            self.thread = Thread(target=self.update_signal_plot, daemon=True)
            self.thread.start()

    def update_signal_plot(self):
        while True:
            for c in range(CHANNELS - 2):
                mic_channel = self.mic_data[:, c]
                self.lines[c].set_ydata(mic_channel)
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception:
                return
