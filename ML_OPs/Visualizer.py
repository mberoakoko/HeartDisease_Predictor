import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Tuple


def plot_confusion_matrix(classifiers, x_test, y_test, n_cols: int = 2, fig_size: Tuple[int, int] = (10, 10)):
    """
    Plots the confusion matrix of various Classifiers
    Args:
    :param x_test: test matrix
    :param y_test:  test target matrix
    :param classifiers: list of classifiers
    :param n_cols: number of columns in subplot
    :param fig_size: size of figure
    :return:
    """

    l = len(classifiers)
    n_rows = int(np.ceil(l/n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)

    for i, pack in enumerate(zip(classifiers, axes.flatten())):
        clf, ax = pack
        y_predict = clf.predict(x_test)
        conf_matrix = confusion_matrix(y_test, y_pred=y_predict)
        sns.heatmap(conf_matrix, ax = ax)

    plt.show()


def plot_roc_auc_curves(classifiers, x_test, y_test):

    plt.subplots(figsize=(12, 12))
    names = [str(type(name)).split(".")[-1][:-2] for name in classifiers]
    for name, classifier in zip(names, classifiers):
        y_predicted_proba = classifier.predict_proba(x_test)[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, y_predicted_proba)
        roc_auc = auc(fpr, tpr)
        name = str(type(classifier)).split(".")[-1][:-2]
        plt.plot(fpr, tpr, lw=3, label=f"{name} ROC area => {roc_auc}")
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curves', fontsize=20)
        plt.legend(loc="lower right")

    plt.show()


