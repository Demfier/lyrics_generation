def main():
    with open('data/processed/ismir_new/2_artists.txt') as f:
        dm_nin = f.readlines()
    with open('data/processed/ismir_new/7-artists-songSegments.txt.LC') as f:
        all_artists = f.readlines()

    new_lines = []
    for line in dm_nin:
        artist, lyrics = line.strip().split('\t')
        new_lines.append('{}\t{}'.format(
            artist, lyrics.lower().encode('ascii', errors='replace').decode('ascii')))

    print(len(new_lines), len(dm_nin))

    for line in all_artists:
        lyrics, artist, _ = line.strip().split(',')
        if artist.split('_', 1)[0] not in ['DepecheMode', 'NineInchNails']:
            new_lines.append('{}\t{}'.format(
                artist, lyrics.lower().encode('ascii', errors='replace').decode('ascii')))

    print(len(new_lines))
    with open('data/processed/ismir_new/clean_lyrics_7_artists_LC.txt', 'w') as f:
        f.write('\n'.join(new_lines))


if __name__ == '__main__':
    main()
