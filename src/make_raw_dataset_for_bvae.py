with open('data/raw/7-artists-songSegments.txt.LC') as f:
    ten_sec_data = f.readlines()
with open('data/raw/complete_lyrics_2artists.txt') as f:
    annotated_data = f.readlines()
print(len(ten_sec_data), len(annotated_data))

new_data = []

for line in ten_sec_data:
    line = line.strip()
    lyrics, spec_id, _ = line.split(',')
    artist = spec_id.split('_', 1)[0]
    if artist.lower() in ['depechemode', 'nineinchnails']:
        continue
    new_data.append('{}\t{}'.format(spec_id, lyrics))

for line in annotated_data:
    line = line.strip()
    spec_id, lyrics = line.split('\t')
    new_data.append('{}\t{}'.format(spec_id[:-4], lyrics))

print(len(new_data))

with open('data/raw/7-artists-songSegments-new.txt', 'w') as f:
    f.write('\n'.join(new_data))
