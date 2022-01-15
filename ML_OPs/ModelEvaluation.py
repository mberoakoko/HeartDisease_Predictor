from sklearn.metrics import roc_curve, accuracy_score, auc, confusion_matrix, recall_score, precision_score, f1_score
from pandas import DataFrame
import numpy as np


def score_summary(classifiers, x_test, y_test):
    """
    Given a list of classiers, this function calculates the accuracy,
    ROC_AUC and Recall and returns the values in a dataframe
    """

    cols = ["Classifier", "Accuracy", "ROC_AUC", "Recall", "Precision", "F1"]
    data_table = DataFrame(columns=cols)
    for classifier in classifiers:
        name = str(type(classifier)).split(".")[-1][:-2]
        predicted = classifier.predict(x_test)
        accuracy = accuracy_score(y_pred=predicted, y_true=y_test)
        pred_proba = classifier.predict_proba(x_test)[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, pred_proba)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_true=y_test, y_pred=predicted)
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        # precision: TP/(TP+FP)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

        # F1 score: TP/(TP+FP)
        f1 = 2 * recall * precision / (recall + precision)
        df = DataFrame([[name, accuracy*100, roc_auc, recall, precision, f1]], columns=cols)
        data_table = data_table.append(df)

    return np.round(data_table.reset_index(drop=True), 2)



