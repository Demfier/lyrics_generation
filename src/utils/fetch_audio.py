import DALI as dali_code

DALI_PATH = '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0/'
AUDIO_PATH = DALI_PATH + 'audio1'
dali_info = dali_code.get_info(DALI_PATH + 'info/DALI_DATA_INFO.gz')
errors = dali_code.get_audio(dali_info, AUDIO_PATH, skip=[], keep=[])
print(errors)
