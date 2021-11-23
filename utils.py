import pyaudio
import numpy as np
import math
from itertools import combinations

from pyroomacoustics.transform import stft

CHUNK = 4096
RATE = 44100
CHANNELS = 8
FORMAT = pyaudio.paInt16
AZIMUTH_RESOLUTION = 1
ELEVATION_RESOLUTION = 10
UI_RESOLUTION = 10


def gcc_phat(x_1, x_2, interp=1):
    """
    Function that will compute the GCC-PHAT
    cross-correlation of two separate audio channels

    Returns:
        A 1-D GCC vector
    """

    n = len(x_1) + len(x_2) - 1
    n += 1 if n % 2 else 0

    # Fourier transforms of the two signals
    X_1 = np.fft.rfft(x_1, n=n)
    X_2 = np.fft.rfft(x_2, n=n)

    # Normalize by the magnitude of FFT - because PHAT
    np.divide(X_1, np.abs(X_1), X_1, where=np.abs(X_1) != 0)
    np.divide(X_2, np.abs(X_2), X_2, where=np.abs(X_2) != 0)

    # GCC-PHAT = [X_1(f)X_2*(f)] / |X_1(f)X_2*(f)|
    # See http://www.xavieranguera.com/phdthesis/node92.html for reference
    CC = X_1 * np.conj(X_2)
    cc = np.fft.irfft(CC, n=n * interp)

    # Maximum delay between a pair of microphones,
    # expressed in a number of samples.
    # 0.09 m is the mic array diameter and
    # 340 m/s is assumed to be the speed of sound.
    max_len = math.ceil(0.09 / 340 * RATE * interp)

    # Trim the cc vector to only include a
    # small number of samples around the origin
    cc = np.concatenate((cc[-max_len:], cc[:max_len + 1]))

    # Return the cross correlation
    return cc


def compute_gcc_matrix(observation, interp=1):
    """
    Creates a GCC matrix, where each row is a vector of GCC 
    between a given pair of microphones.
    """

    mic_pairs = combinations(range(6), r=2)

    # Initialize a transformed observation, that will be populated with GCC vectors
    # of the observation
    transformed_observation = []

    # Compute GCC for every pair of microphones
    for mic_1, mic_2 in mic_pairs:
        x_1 = observation[:, mic_1]
        x_2 = observation[:, mic_2]

        gcc = gcc_phat(x_1, x_2, interp=interp)

        # Add the GCC vector to the GCC matrix
        transformed_observation.append(gcc)

    return transformed_observation


def compute_stft_matrix(observation, nfft=256):
    """
    Creates a STFT matrix using microphone data from 6 channels.
    """

    # Default value for overlap
    step = nfft // 2

    # Calculate multidimensional STFT and return
    transformed_observation = stft.analysis(observation, L=nfft, hop=step)
    return np.transpose(transformed_observation, axes=[2, 1, 0])
