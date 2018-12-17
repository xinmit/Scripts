import sys
import subprocess
import librosa as lr
import os
import glob
from pathlib import Path
from random import randint
import  scipy.io.wavfile

import numpy as np
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from addNoiseReverb import add_convolve, add_noise
from itertools import repeat

noise_files = []
rir_files=[]
snr_lvl = 20
sample_rate = 16000

def gen_noise(clean_wav, noise_wav, snr):

    x_s , x_sr = lr.core.load(clean_wav, sr=sample_rate,mono=True)
    z_s, z_sr = lr.core.load(noise_wav,sr=sample_rate, mono=True)
    x_n = add_noise(x_s, z_s, snr)

    return x_n


def gen_reverb(clean_wav, reverb_wav):
    x_s , x_sr = lr.core.load(clean_wav, sr=sample_rate,mono=True)
    z_s, z_sr = lr.core.load(reverb_wav,sr=sample_rate, mono=True)
    x_ir = add_convolve(x_s, z_s)

    return x_ir



def read_file(file_list):
    
    list = []

    with open(file_list, "r") as fp:
        for line in fp:
            line = line.rstrip()
            list.append(line)
    return list


def gen_noise_file(clean_wav,count):
    clean_wav_dir = os.path.dirname(clean_wav)
    spk_name = clean_wav_dir.split('/')[-1]
    noise_spk_name = spk_name + "n{}".format(count)
    noise_wav_dir = os.path.join(str(Path(clean_wav_dir).parent), noise_spk_name)

    noise_wav_file = clean_wav.replace(spk_name, noise_spk_name)
    os.makedirs(noise_wav_dir, mode=0o755, exist_ok=True)


    #gen the noisefile
    noise_wav = noise_files[randint(0, len(noise_files)-1)]
 
    x_n = gen_noise(clean_wav, noise_wav,snr=snr_lvl)
    x_n = [ x * pow(2,15) for x in x_n]

    #save into wav
    scipy.io.wavfile.write(noise_wav_file, sample_rate, np.asarray(x_n, dtype=np.int16))

if __name__ == '__main__':
    
    noise_file_list  = "/home/rxia/Documents/data/ASR/noise_list/noise_file.lst"
    ir_file_list = "/home/rxia/Documents/data/ASR/noise_list/rir_file.lst"
    clean_wav_file_list = "/home/rxia/Documents/data/kyat_seg/wk_train.lst";
    noise_files = read_file(noise_file_list)
    clean_wav_files = read_file(clean_wav_file_list)

    cp_num = 5



    for i in range(1,cp_num+1):
        with ProcessPoolExecutor(max_workers=4) as executor:
            executor.map(gen_noise_file, clean_wav_files, repeat(i))