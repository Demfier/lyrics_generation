import os
import librosa
from librosa.display import specshow

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

OGG_DIR = 'data/raw/DALI_v1.0/ogg_audio/'
SPEC_DIR = 'data/processed/spectrograms/'

for a in tqdm(os.listdir(OGG_DIR)):
    y, _ = librosa.load('{}{}'.format(OGG_DIR, a))
    s = librosa.feature.melspectrogram(y)
    specshow(librosa.power_to_db(s, ref=np.max), fmax=8000)
    plt.tight_layout()
    plt.savefig('{}{}.png'.format(SPEC_DIR, a.split('.')[0]), bbox_inches='tight')
