import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap='Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if title is None:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    classes = np.asarray(classes)[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    height = np.arange(cm.shape[0])
    width = np.arange(cm.shape[1])

    ax.set(xticks=width,
           yticks=height,
           xticklabels=classes,
           yticklabels=classes,
           xlabel='Predicted label',
           ylabel='True label',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(height, width):
        ax.text(j, i, format(cm[i, j], fmt),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

    if matplotlib.__version__ == '3.1.1':
        high, low = ax.get_ylim()
        ax.set_ylim(high + 0.5, low - 0.5)

    fig.tight_layout()
    return ax
