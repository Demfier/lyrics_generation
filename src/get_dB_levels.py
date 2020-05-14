import os
import math
import librosa
import numpy as np
import scipy


def get_dB(wav_path):
    N = 2**15 - 1 if 'Ghosts' in wav_path else 2**31 - 1
    samplerate, data = scipy.io.wavfile.read(wav_path)
    return 20*(math.log(np.max(data.flatten())/N, 10))


if __name__ == '__main__':
    # wav_dir = 'data/raw/NineInchNails_with-teeth_37/'
    wav_dir = '/collection/ovechtom/ismir-paper-experiments/audio-clip-conditioned/10-sec-wav/'
    dB_level = ''
    for f in os.listdir(wav_dir):
        dB = get_dB('{}{}'.format(wav_dir, f))
        dB_level += '{},{}\n'.format(f, dB)

    with open('dB_level.csv', 'w') as f:
        f.write(dB_level)
