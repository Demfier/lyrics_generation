import gensim
import pickle
from models.config import model_config as config


class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, file_path):
        super(Dataset, self).__init__()
        self.file_path = file_path

    def __iter__(self):
        # input is a pickle file
        for line in pickle.load(open(self.file_path, 'rb')):
            yield line[0].split()


def main(file_path):
    """
    Trains a word2vec model on the given dataset
    W2V model learns embeddings of size, ignores words with frequency < 5,
    and parallalizes using 4 workers
    """
    sentences = Dataset(file_path)
    model = gensim.models.Word2Vec(sentences, size=300, workers=4, iter=50)
    model.save('data/processed/english_w2v.pkl')
    print('Trained word2vec on the corpus at {}'.format(file_path))


if __name__ == '__main__':
    main('data/processed/{}/combined_dataset.pkl'.format(config['model_code']))
