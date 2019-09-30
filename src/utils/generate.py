import os
import re
import sys
import h5py
import math
import json
import torch
import gensim
import pickle
import random
import skimage
import librosa
import itertools
import unicodedata
import numpy as np
import pandas as pd
import DALI as dali_code
from itertools import zip_longest
from pydub import AudioSegment
from librosa.display import specshow
import matplotlib.pyplot as plt


def run(config):
    print('Loading DALI')
    dali_data = dali_code.get_the_DALI_dataset(config['dali_path'],
                                               skip=[], keep=[])
    dataset = []
    spec_path = '{}check/'.format(config['data_dir'])
    file_ids = [p.split('.')[0] for p in os.listdir(spec_path)]
    #print(dali_data)
    p = 0
    b = 0
    print(len(file_ids))
    for f_id in file_ids:
        sample = {}
        info = dali_data[f_id].info
        #if 'genres' in info['metadata'] and len(info['metadata']['genres']) == 1 and info['metadata']['language'] == config['filter_lang'] and info['metadata']['genres'][0] == 'Metal' :
        p += 1
        #print(info['metadata']['genres'])
        #print(info)
        #print(dali_data[f_id].annotations['annot']['lines'])
        if 'genres' in info['metadata'] and len(info['metadata']['genres']) > 0 and info['metadata']['language'] == config['filter_lang'] and (info['metadata']['genres'][0] == 'Pop'):
            b += 1
            print(b)
            a = dali_data[f_id].annotations['annot']['lines']
            print(p)
            #print(a)
            if config['generate_spectrograms']:
                for i in range(len(a)):
                    generate_spectrograms(config['dali_audio_split'] + f_id + "_" + str(i) +  ".ogg", config['spectrograms'] + f_id + "_" + str(i) + ".png")
            else:
                if len(a) == 0:
                    # if the song has no lyrics then we dont need it
                    continue
                if len(a) == 1:
                    audioSegment = AudioSegment.from_ogg(config['dali_audio'] + f_id + ".ogg")
                    split_audio_file(config['dali_audio'] + f_id + ".ogg", config['dali_audio_split'] + f_id + "_" + str(0) +  ".ogg", 0, len(audioSegment))
                    with open(config['file_name'], "a") as file:
                        file.write(a[0]['text'] + '\t' + config['dali_audio_split'] + f_id + "_" + str(0) +  ".ogg" + '\n')
                    file.close()
                    # if the song has just one line in the lyrics then the full song should go
                for i in range(len(a)):
                    if i > 0:
                        if i == 1:
                            t1 = 0
                            #t1 = previous_t2
                            t2 = (a[i-1]['time'][1] + a[i]['time'][0])*0.5
                            split_audio_file(config['dali_audio'] + f_id + ".ogg", config['dali_audio_split'] + f_id + "_" + str(i-1) +  ".ogg", t1, t2*1000)
                            #print(a[i-1]['text'])
                            with open(config['file_name'], "a") as file:
                                file.write(a[i-1]['text'] + '\t' + config['dali_audio_split'] + f_id + "_" + str(i-1) +  ".ogg" + '\n')
                            file.close()
                            #print(t1)
                            #print(t2)
                            previous_t2 = t2
                        elif i < len(a)-1:
                            t1 = previous_t2
                            t2 = (a[i-1]['time'][1] + a[i]['time'][0])*0.5
                            split_audio_file(config['dali_audio'] + f_id + ".ogg", config['dali_audio_split'] + f_id + "_" + str(i-1) +  ".ogg", t1*1000, t2*1000)
                            with open(config['file_name'], "a") as file:
                                file.write(a[i-1]['text'] + '\t' + config['dali_audio_split'] + f_id + "_" + str(i-1) +  ".ogg" + '\n')
                            file.close()
                            previous_t2 = t2
                            #print(a[i-1]['text'])
                            #print(t1)
                            #print(t2)
                        else:
                            t1 = previous_t2
                            t2 = (a[i-1]['time'][1] + a[i]['time'][0])*0.5
                            split_audio_file(config['dali_audio'] + f_id + ".ogg", config['dali_audio_split'] + f_id + "_" + str(i-1) +  ".ogg", t1*1000, t2*1000)
                            with open(config['file_name'], "a") as file:
                                file.write(a[i-1]['text'] + '\t' + config['dali_audio_split'] + f_id + "_" + str(i-1) +  ".ogg" + '\n')
                            file.close()
                            #print(a[i-1]['text'])
                            #print(t1)
                            #print(t2)
                            t1 = t2
                            audioSegment = AudioSegment.from_ogg(config['dali_audio'] + f_id + ".ogg")
                            t2 = len(audioSegment)
                            split_audio_file(config['dali_audio'] + f_id + ".ogg", config['dali_audio_split'] + f_id + "_" + str(i) +  ".ogg", t1*1000, t2)
                            with open(config['file_name'], "a") as file:
                                file.write(a[i]['text'] + '\t' + config['dali_audio_split'] + f_id + "_" + str(i) +  ".ogg" + '\n')
                            file.close()
                            print(a[i]['text'])
                            print(t1)
                            print(t2)
                #previous = a[i]['text']
                #print(a[i]['time'])
        # list of dictionaries
        #p += 1'''
        #print(f_id)
        #print(info)
        #sample['lyrics'] = dali_data[f_id].annotations['annot']['lines']
        #sample['mel_spec'] = read_spectrogram('{}{}.png'.format(spec_path, f_id))
        #dataset.append(sample)
    #print(p)
    print('Done')
    #return dataset

def split_audio_file(read_dest, write_dest, t1, t2):
    t1 = t1 #Works in milliseconds
    t2 = t2
    newAudio = AudioSegment.from_ogg(read_dest)
    newAudio = newAudio[t1:t2]
    newAudio.export(write_dest, format="ogg")


def generate_spectrograms(read_dest, write_dest):
    #a = data[i]
    try:
        y, _ = librosa.load(read_dest)
        #y, _ = librosa.load('{}{}'.format(OGG_DIR, a))
        plt.figure(0)
        plt.axis('off')
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        s = librosa.feature.melspectrogram(y)
        specshow(librosa.power_to_db(s, ref=np.max), fmax=8000)
        plt.tight_layout()
        plt.savefig(write_dest, bbox_inches='tight')
        plt.close(0)
    except Exception as e:
        print(e)
