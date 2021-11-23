import os
import pathlib

import tensorflow as tf

from predictor import Predictor, get_mic_data, get_input_matrix, get_model_details
from utils import *


def init_models():
    print('Loading models...')
    # Load the TFLite model and allocate tensors.
    base_dir = pathlib.Path(__file__).parent.parent.absolute()
    az_model_file = os.path.join(base_dir, 'models', 'best_multi_source_model.tflite')
    az_interpreter = tf.lite.Interpreter(model_path=az_model_file)

    print('Allocating tensors...')
    az_interpreter.allocate_tensors()
    print('Tensors allocated.\n')

    # Get input and output tensors for the model
    az_input_details, az_input_shape, az_output_details, az_output_shape = get_model_details(az_interpreter)

    print('Azimuth model input tensor: ' + str(az_input_shape))
    print('Azimuth model output tensor: ' + str(az_output_shape))
    print('\nModels ready. Press Start to begin inference.\n')

    return az_interpreter, az_input_details, az_output_details


class MultiSourcePredictor(Predictor):
    def __init__(self, lines, fig, thresh=50, max_silence_frames=10):
        super().__init__(lines, fig, thresh, max_silence_frames)
        self.az_current_predictions = []

        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
            frames_per_buffer=CHUNK, stream_callback=self.callback
        )

        self.stream.start_stream()
        self.az_interpreter, self.az_input_details, self.az_output_details = init_models()

    def callback(self, in_data, frame_count, time_info, status):
        data, mic_data = get_mic_data(in_data)

        if self.is_active:
            self.mic_data = mic_data

        if abs(np.max(mic_data)) > self.thresh and self.is_active:
            self.az_current_predictions = self.get_prediction_from_model(mic_data)
        else:
            if self.silent_frames == self.max_silence_frames:
                self.silent_frames = 0
                self.az_current_predictions = []
            self.silent_frames += 1
        if self.is_active:
            self.output_predictions()

        return data, pyaudio.paContinue

    def get_prediction_from_model(self, mic_data):
        input_data = get_input_matrix(mic_data)

        # Set input and run azimuth interpreter
        self.az_interpreter.set_tensor(self.az_input_details['index'], input_data)
        self.az_interpreter.invoke()
        az_output_data = self.az_interpreter.get_tensor(self.az_output_details['index'])
        return az_output_data[0]

    def output_predictions(self):
        predictions = [(angle * AZIMUTH_RESOLUTION, round(conf, 3))
                       for angle, conf in enumerate(self.az_current_predictions) if conf > 0.5]
        if len(predictions):
            print(predictions)
        else:
            print('[No prediction]')
