import os
import pathlib
import time

import tensorflow as tf

from predictor import Predictor, get_mic_data, get_input_matrix, get_model_details
from utils import *
import pyroomacoustics as pra


def init_models():
    print('Loading models...')
    # Load the TFLite model and allocate tensors.
    base_dir = pathlib.Path(__file__).parent.parent.absolute()
    az_model_file = os.path.join(base_dir, 'models', 'best_super_azimuth_model.tflite')
    el_model_file = os.path.join(base_dir, 'models', 'elevation_model.tflite')
    az_interpreter = tf.lite.Interpreter(model_path=az_model_file)
    el_interpreter = tf.lite.Interpreter(model_path=el_model_file)

    print('Allocating tensors...')
    az_interpreter.allocate_tensors()
    el_interpreter.allocate_tensors()
    print('Tensors allocated.\n')

    # Get input and output tensors for both models
    az_input_details, az_input_shape, az_output_details, az_output_shape = get_model_details(az_interpreter)
    el_input_details, el_input_shape, el_output_details, el_output_shape = get_model_details(el_interpreter)

    print('Azimuth model input tensor: ' + str(az_input_shape))
    print('Azimuth model output tensor: ' + str(az_output_shape))
    print('Elevation model input tensor: ' + str(el_input_shape))
    print('Elevation model output tensor: ' + str(el_output_shape))
    print('\nModels ready. Press Start to begin inference.\n')

    return az_interpreter, az_input_details, az_output_details, el_interpreter, el_input_details, el_output_details


class SingleSourcePredictor(Predictor):
    def __init__(self, CNN, lines, fig, thresh=50, max_silence_frames=10):
        super().__init__(lines, fig, thresh, max_silence_frames)
        self.az_current_prediction = None
        self.el_current_prediction = None
        self.az_confidences = np.zeros(360 // AZIMUTH_RESOLUTION)
        self.CNN = CNN

        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
            frames_per_buffer=CHUNK, stream_callback=self.callback
        )

        self.cnn_exec_times = []
        self.music_exec_times = []

        self.stream.start_stream()
        self.az_interpreter, self.az_input_details, self.az_output_details, \
            self.el_interpreter, self.el_input_details, self.el_output_details = init_models()

    def callback(self, in_data, frame_count, time_info, status):
        data, mic_data = get_mic_data(in_data)

        if self.is_active:
            self.mic_data = mic_data

        if abs(np.max(mic_data)) > self.thresh and self.is_active:
            input_data = get_input_matrix(mic_data)
            if self.CNN.get():
                self.az_current_prediction = self.get_azimuth_prediction(input_data)
            else:
                self.az_current_prediction = self.run_music(mic_data)
            self.el_current_prediction = self.get_elevation_prediction(input_data)
        else:
            if self.silent_frames == self.max_silence_frames:
                self.silent_frames = 0
                self.az_current_prediction = None
                self.az_confidences = np.zeros(360 // AZIMUTH_RESOLUTION)
                self.el_current_prediction = None
            self.silent_frames += 1
        if self.is_active:
            self.output_predictions()

        return data, pyaudio.paContinue

    def run_music(self, mic_data):
        start_time = time.time()
        stft_data = compute_stft_matrix(mic_data)
        az_current_prediction = self.get_music_prediction(stft_data)
        self.music_exec_times.append(time.time() - start_time)
        return az_current_prediction

    def get_azimuth_prediction(self, input_data):
        # Set input and run azimuth interpreter
        if len(self.az_input_details['shape']) == 4:
            input_data = input_data[..., np.newaxis]

        if self.az_input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = self.az_input_details["quantization"]
            input_data = input_data / input_scale + input_zero_point

        start_time = time.time()
        self.az_interpreter.set_tensor(self.az_input_details['index'],
                                       input_data.astype(self.az_input_details['dtype']))
        self.az_interpreter.invoke()

        az_output_data = self.az_interpreter.get_tensor(self.az_output_details['index'])[0]
        execution_time = time.time() - start_time
        self.cnn_exec_times.append(execution_time)

        if self.az_output_details['dtype'] == np.uint8:
            output_scale, _ = self.az_output_details["quantization"]
            az_output_data = az_output_data * output_scale

        self.az_confidences = az_output_data

        # Get the predicted azimuth as argument of the max probability
        az_prediction, az_confidence = np.argmax(self.az_confidences) * AZIMUTH_RESOLUTION, np.max(self.az_confidences)
        return az_prediction, az_confidence

    def get_music_prediction(self, input_data):
        mic_center = [2, 2]
        mic_height = 1

        # Radius constant, will always be the same for MiniDSP array
        mic_radius = 0.045
        R = pra.circular_2D_array(center=mic_center, M=6, phi0=0, radius=mic_radius)
        R = np.vstack((R, [mic_height] * 6))

        # Run MUSIC algorithm for DOA
        doa = pra.doa.MUSIC(R, RATE, 256, n_grid=(360 // AZIMUTH_RESOLUTION))
        doa.locate_sources(input_data)

        prediction = round((doa.azimuth_recon[0] * 180 / math.pi))
        self.az_confidences = np.zeros(360 // AZIMUTH_RESOLUTION)
        self.az_confidences[prediction] = 1

        return round((doa.azimuth_recon[0] * 180 / math.pi)), 1

    def get_elevation_prediction(self, input_data):
        # Set input and run elevation interpreter
        self.el_interpreter.set_tensor(self.el_input_details['index'], input_data)
        self.el_interpreter.invoke()
        el_output_data = self.el_interpreter.get_tensor(self.el_output_details['index'])

        # Get the predicted elevation as argument of the max probability
        el_prediction, el_confidence = np.argmax(el_output_data[0]) * ELEVATION_RESOLUTION, np.max(el_output_data[0])

        return el_prediction, el_confidence

    def output_predictions(self):
        if self.az_current_prediction is not None:
            az_pred, az_conf = self.az_current_prediction
            el_pred, el_conf = self.el_current_prediction
            az_conf = round(az_conf * 100, 1)
            el_conf = round(el_conf * 100, 1)
            print('Azimuth: {:>3} degrees [{:>5}%]'.format(az_pred, az_conf), end=' | ')
            print('Elevation: {:>3} degrees [{:>5}%]'.format(el_pred, el_conf))
        else:
            print('{:<63}'.format('[No prediction]'))
