from scipy.signal import fftconvolve
import scipy.io.wavfile as sw
from datetime import datetime
import numpy as np
import random


def add_noise(x,z,snr):
    """

    :param x: Input speech signal
    :param z: Input noise signal
    :param snr: expected SNR level
    :return: corrupted noise signal
    """

    while z.shape[0] < x.shape[0]:
        z = np.concatenate((z,z), axis=0)


    random.seed(datetime.now())
    ran_index = random.randint(0,z.shape[0] - x.shape[0])
    z =z[ran_index : ran_index + x.shape[0]]

    rms_z = np.sqrt( np.mean(np.power(z,2)))
    rms_x = np.sqrt( np.mean(np.power(x,2)))

    snr_linear = 10 ** (0.05 * snr)
    noise_factor = rms_x / rms_z /snr_linear;

    
    y = x + z * noise_factor
    y = 0.8 / np.max(np.abs(y)) * y
    return y

def add_convolve(x, h):
    """

    :param x: Input speech signal
    :param h: Impuluse response
    :return: convolved signal
    """

    y = fftconvolve(x,h)
    y = y[:x.shape[0]]

    y = 0.8 / np.max(np.abs(y)) * y

    return y




