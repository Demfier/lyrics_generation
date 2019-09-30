import nltk
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance


def plot_confusion_matrix(targets, predictions, classes,
                          epoch, model_code, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    cm = confusion_matrix(targets, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = 100*np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        v = cm[i, j]
        if normalize:
            v = np.round(v, decimals=1)
        plt.text(j, i, v,
                 horizontalalignment="center",
                 color="white" if v > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not os.path.exists('reports/figures/{}'.format(model_code)):
        os.mkdir('reports/figures/{}'.format(model_code))
    plt.savefig('reports/figures/{}/cm_{}'.format(model_code, epoch))
    plt.close()


def calculate_bleu_scores(hypothesis, references):
    bleu_scores = {}
    for i in range(1, 5):
        w = 1.0 / i
        weights = [w] * i
        bleu_scores['bleu{}'.format(str(i))] = 100 * nltk.translate.bleu_score\
            .corpus_bleu(references, hypothesis, weights=weights)
    return bleu_scores
