import re
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

with open('data/processed/instrumental/image_mu_testing_ISMIR.pkl', 'rb') as f:
    instrumental_mu = pickle.load(f)

with open('data/processed/instrumental/image_mu_combined_ISMIR.pkl', 'rb') as f:
    combined_mu = pickle.load(f)

with open('data/processed/instrumental/instrumental_songs_albums.json', 'r') as f:
    songs_albums = json.load(f)


same_song_prop = []
same_artist_prop = []
same_album_prop = []

n = 100

instrumental_song_ids = list(instrumental_mu.keys())

pbar = tqdm(instrumental_song_ids)

for i in pbar:
    # print(i)
    if len(i.split(' ')) == 1:
        modified_i = i.replace('-', ' ')
    else:
        modified_i = i
    song_id = modified_i.split(' ', 1)[1]
    song = song_id.split('_')[0]

    album = 0
    artist = ''
    if 'Ghosts' in song:
        album = 0
        artist = 'NineInchNails'
    else:
        for a_id in songs_albums:
            if song in songs_albums[a_id]['songs']:
                album = int(a_id)
                artist = songs_albums[a_id]['artist']
                break

    instr_z = instrumental_mu[i]
    scored_z = []
    for k, v in combined_mu.items():
        score = cosine_similarity(instr_z, v)[0][0]
        if k != i:
            scored_z.append((score, k))
    scored_z = sorted(scored_z, reverse=True)
    scored_z = scored_z[:n]

    # print(scored_z)
    same_songs = 0
    same_artists = 0
    same_albums = 0

    for (s, scored_i) in scored_z:
        if scored_i in instrumental_song_ids:
            if len(scored_i.split(' ')) == 1:
                curr_modified_i = scored_i.replace('-', ' ')
            else:
                curr_modified_i = scored_i
            curr_song_id = curr_modified_i.split(' ', 1)[1]
            curr_song = curr_song_id.split('_')[0]
        else:
            curr_artist, curr_song_id = scored_i.split('_', 1)
            if curr_song_id.endswith('.png'):
                curr_song = curr_song_id.split('_')[0].replace('-', ' ')
            else:
                curr_song = curr_song_id[:-3].replace('-', ' ')

        curr_album = 0
        if 'Ghosts' in curr_song:
            curr_album = 0
            curr_artist = 'NineInchNails'
        else:
            found_album = False
            for a_id in songs_albums:
                if curr_song.lower() in [s.lower() for s in songs_albums[a_id]['songs']]:
                    curr_album = int(a_id)
                    curr_artist = songs_albums[a_id]['artist']
                    found_album = True
                    break
            if not found_album:
                curr_album = 4

        if curr_song.lower() == song.lower():
            same_songs += 1

        # print(song, curr_song, song_id)
        # print(artist, curr_artist)
        # print(album, curr_album)

        if curr_artist == artist:
            same_artists += 1

        if curr_album == album:
            same_albums += 1

    same_songs /= float(n)
    same_artists /= float(n)
    same_albums /= float(n)

    same_song_prop.append(same_songs)
    same_artist_prop.append(same_artists)
    same_album_prop.append(same_albums)

    pbar.set_postfix(
        song_prop=np.mean(same_song_prop),
        artist_prop=np.mean(same_artist_prop),
        album_prop=np.mean(same_album_prop),
        )

pbar.close()

print(np.mean(same_song_prop))
print(np.mean(same_artist_prop))
print(np.mean(same_album_prop))
