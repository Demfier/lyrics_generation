import kenlm
import numpy as np

LANGUAGE_MODEL_PATH = ''
FILE_PATH = ''

with open(FILE_PATH, 'r') as f:
    sentences = f.readlines()

language_model = kenlm.Model(LANGUAGE_MODEL_PATH)

scores = []
for s in sentences:
    scores.append(model.score(s))

print(np.mean(scores))
