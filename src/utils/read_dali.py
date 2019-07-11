from time import time
import DALI as dali_code

s = time()
DALI_PATH = '/home/gsahu/code/lyrics_generation/data/raw/DALI_v1.0/'
dali_data = dali_code.get_the_DALI_dataset(DALI_PATH, skip=[], keep=[])
f_id = dali_data['7b59545605de4c7cb58f2f24ab89895e']

print('Reading Dali info')
dali_info = dali_code.get_info(DALI_PATH + 'info/DALI_DATA_INFO.gz')
path_audio = '/home/gsahu/code/lyrics_generation/data/raw/'
print('Getting audio files')
errors = dali_code.get_audio(dali_info, path_audio, skip=[], keep=[])

f_id.horizontal2vertical()
print(f_id.info)
a = f_id.annotations
my_annot = a['annot']['hierarchical']
lines_1paragraph = my_annot[0]['text']
words_1line_1paragraph = lines_1paragraph[0]['text']
print(words_1line_1paragraph[0])

print(a['type'])
# print(a['annot_param'])

# print('Sample GT:\nnotes: {}\nwords: {}\nlines: {}\npara: {}'.format(a['annot']['notes'][0], a['annot']['words'][0], a['annot']['lines'][0], a['annot']['paragraphs'][0]))
