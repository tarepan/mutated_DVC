'''
This code is about dataset.
There is two dataset in here. You can chose both dataset.

WaveDataset takes a wave file.
This dataset convert from a wave file to input datas.
The process what has many times FFT is repeated, so this use many CPU resouces.

PreEncodedDataset takes a numpy file that is pre-encoded datas.
You can get the pre-encoded file by runnning this .py file, or pre_encode method.
'''

import numpy as np
import os
import random
import scipy.io.wavfile as wav
import chainer
import pickle
from nets.models import padding

def load(path):
    bps, data = wav.read(path)
    if len(data.shape) != 1:
        data = data[:,0] + data[:,1]
    return bps, data

def save(path, bps, data):
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    data = np.reshape(data, -1)
    wav.write(path, bps, data)

def find_wav(path):
    name = os.listdir(path)
    dst = []
    for n in name:
        if n[-4:] == '.wav':
            dst.append(path + "/" + n)
    return dst

scale = 9
bias = -6.2

height = 64
sride = 64
dif = height*sride

class WaveDataset(chainer.dataset.DatasetMixin):
    def __init__(self, wave, dataset_len, test):
        self.wave = np.array(load(wave)[1], dtype=float)
        self.max = len(self.wave)-dif-sride*(3+padding*2)
        self.length = dataset_len
        if dataset_len <= 0:
            self.length = self.max // dif
        self.window = np.hanning(254)
        self.test = test

    def __len__(self):
        return self.length
    
    def get_example(self, i):
        if self.test:
            p = i * dif
        else:
            while True:
                p = random.randint(0, self.max)
                if np.max(self.wave[p:p+dif]) > 1000:
                    break
        return wave2input_image(self.wave, self.window, p, padding)

class PreEncodedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, encoded_path, dataset_len, test):
        self.images = np.load(encoded_path)
        self.max = self.images.shape[1]-height - padding*2
        self.length = dataset_len
        if dataset_len <= 0:
            self.length = self.max // height
        self.test = test

    def __len__(self):
        return self.length
    
    def get_example(self, i):
        if self.test:
            p = i * height
        else:
            while True:
                p = random.randint(0, self.max)
                if np.max(self.images[:,p:p+height,:]) > 0.4:
                    break
        return np.copy(self.images[:,p:p+height+padding*2,:])


from scipy.interpolate import interp2d
import numpy as np
from librosa.core import hz_to_mel as hz2mel
from librosa.core import mel_to_hz as mel2hz
def linear2mel(spectrogram, *, freq_min, freq_max):
    """
    Convert linear-frequency into mel-frequency w/o dimension compression (with linear interpolation in linear-frequency domain)
    Args:
        spectrogram (numpy.ndarray 2D): magnitude, IF and any other mel-compatible spectrogram
    Returns:
        numpy.ndarray 2D: mel-nized spectrogram
    """
    linear_freq = np.linspace(start=freq_min, stop=freq_max, num=spectrogram.shape[0], endpoint=True)
    time = range(spectrogram.shape[1])
    melnizer = interp2d(time, linear_freq, spectrogram)

    even_spaced_mel = np.linspace(start=hz2mel(freq_min, htk=True), stop=hz2mel(freq_max, htk=True), num=spectrogram.shape[0], endpoint=True)
    mel_in_freq = [mel2hz(mel, htk=True) for mel in even_spaced_mel]
    mel_spectrogram = melnizer(time, mel_in_freq)
    return mel_spectrogram

def mel2linear(melspectrogram, *, freq_min, freq_max):
    """
    Convert mel-frequency into linear-frequency w/o dimension compression (with linear interpolation in linear-frequency domain)
    Args:
        melspectrogram (numpy.ndarray 2D): magnitude, IF and any other melspectrogram
    Returns:
        numpy.ndarray 2D: linear-nized spectrogram
    """
    time = range(melspectrogram.shape[1])
    even_spaced_mel = np.linspace(start=hz2mel(freq_min, htk=True), stop=hz2mel(freq_max, htk=True), num=melspectrogram.shape[0], endpoint=True)
    mel_in_freq = [mel2hz(mel, htk=True) for mel in even_spaced_mel]
    linearnizer = interp2d(time, mel_in_freq, melspectrogram)

    linear_freq = np.linspace(start=freq_min, stop=freq_max, num=melspectrogram.shape[0], endpoint=True)
    spectrogram = linearnizer(time, linear_freq)
    return spectrogram


def wave2input_image(wave, window, pos=0, pad=0):
    wave_image = np.hstack([wave[pos+i*sride:pos+(i+pad*2)*sride+dif].reshape(height+pad*2, sride) for i in range(256//sride)])[:,:254]
    wave_image *= window
    spectrum_image = np.fft.fft(wave_image, axis=1)

    # mel-nize
    input_image = linear2mel(np.abs(spectrum_image[:,:128].T, dtype=np.float32), freq_min=0, freq_max=16000).T
    input_image = input_image.reshape(1, height+pad*2, 128)

    np.clip(input_image, 1000, None, out=input_image)
    np.log(input_image, out=input_image)
    input_image += bias
    input_image /= scale

    if np.max(input_image) > 0.95:
        print('input image max bigger than 0.95', np.max(input_image))
    if np.min(input_image) < 0.05:
        print('input image min smaller than 0.05', np.min(input_image))

    return input_image

def reverse(output_image):
    src = output_image[0,padding:-padding,:]
    src[src > 1] = 1
    src *= scale
    src -= bias
    np.abs(src, out=src)
    np.exp(src, out=src)

    # linear-nize
    src = mel2linear(src.T, freq_min=0, freq_max=16000).T
    src[src < 1000] = 1

    mil = np.array(src[:,1:127][:,::-1])
    src = np.concatenate([src, mil], 1)

    return src.astype(complex)


def pre_encode():
    import tqdm

    path = input('enter wave path...')
    ds = WaveDataset(path, -1, True)
    num = ds.max // dif

    imgs = [ds.get_example(i) for i in tqdm.tqdm(range(num))]    
    dst = np.concatenate(imgs, axis=1)
    print(dst.shape)

    np.save(path[:-3]+'npy', dst)
    print('encoded file saved at', path[:-3]+'npy')


if __name__ == "__main__":
    pre_encode()
