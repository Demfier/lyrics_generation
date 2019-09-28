import os

wav_path = 'data/raw/DALI_v1.0/audio/'
aiff_path = 'data/raw/DALI_v1.0/ogg_audio/'
wav_files = os.listdir(wav_path)

for w in wav_files:
	os.system('ffmpeg -i {}{} {}{}.ogg'.format(wav_path, w, aiff_path, w.split('.')[0]))
