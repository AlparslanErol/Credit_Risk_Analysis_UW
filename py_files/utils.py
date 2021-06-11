#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import plot_roc_curve, roc_auc_score, precision_recall_curve, make_scorer
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, roc_curve, auc, classification_report, f1_score


# method for visualize roc curve by using evaluation method above
def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""

    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14, 10))

    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)


# method for calculating area under the curve (AUC) score.
def compute_roc_auc(index, X, y, clf):
    y_predict = clf.predict_proba(X.loc[X.index.intersection(index)])[:, 1]
    fpr, tpr, thresholds = roc_curve(y.loc[y.index.intersection(index)], np.array(y_predict))
    auc_score = auc(fpr, tpr)
    y_true = y.loc[y.index.intersection(index)]
    return fpr, tpr, auc_score, y_true, y_predict


# method for calculating area under the curve (AUC) score.
def compute_roc_auc_(X, y, clf):
    y_predict = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    auc_score = auc(fpr, tpr)
    y_true = y
    return fpr, tpr, auc_score, y_true, y_predict, y_prob


# method for evaluating models with StratifiedKFold by using above methods.
def roc(X_train, X_test, y_train, y_test, clf):
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    results = pd.DataFrame(columns=['training_score', 'test_score'])
    fprs, tprs, scores, predictions, probabilities = [], [], [], [], []
     #
    for (train, test), i in zip(cv.split(X_train, y_train), range(5)):
        clf.fit(X_train.loc[X_train.index.intersection(train)], y_train.loc[y_train.index.intersection(train)])
        _, _, auc_score_train, _, _, = compute_roc_auc(train, X_train, y_train, clf)
        fpr, tpr, auc_score, y_true, y_predict, y_prob = compute_roc_auc_(X_test, y_test, clf)
        probabilities.append(y_prob)
        predictions.append(y_predict)
        scores.append((auc_score_train, auc_score))
        fprs.append(fpr)
        tprs.append(tpr)
        print("\nEvaluation results for fold {} ..: ".format(i+1))
        print(evaluate(y_true, y_predict, print_cm=True))

    plot_roc_curve(fprs, tprs);
    print(pd.DataFrame(scores, columns=['AUC Train', 'AUC Test']))
    y_pred = pd.DataFrame(predictions).mean().round()
    y_pred = y_pred.values.reshape((y_test.shape[0],)).astype(int)
    y_prob = pd.DataFrame(probabilities).mean()
    eval_(y_test, y_pred, y_prob)
    return y_pred


# method for evaluating classification metrics like accuracy, precision, recall and f1 score
def evaluate(y_true, y_pred, print_cm=True):
    # calculate and display confusion matrix
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if print_cm:
        print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels\n')
        print(cm)
        print('\n')

    # calculate precision, recall, and F1 score
    accuracy = float(np.trace(cm)) / np.sum(cm)
    precision = precision_score(y_true, y_pred, average=None, labels=labels)[1]
    recall = recall_score(y_true, y_pred, average=None, labels=labels)[1]
    f1 = 2 * precision * recall / (precision + recall)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1 score:", f1)


# evaluation method for test data with constructing confusion matrix (cm)
def eval_(y_test, pred_y, prob_y):
    # Is our model still predicting just one class?
    print("\nUnique values are..:", np.unique(pred_y))

    # How's our accuracy?
    print("Accurcy for test data..:", accuracy_score(y_test, pred_y))

    # ROC Score
    print("ROC for test data..:", roc_auc_score(y_test, prob_y))

    print("\nHere is the Confusion Matrix...")
    # confusion matrix
    data = confusion_matrix(y_test, pred_y)
    df_cm = pd.DataFrame(data, columns=['No Default', 'Default'], index=['No Default', 'Default'])
    df_cm.index.name = 'True Label'
    df_cm.columns.name = 'Predicted Label'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap='viridis', annot=True, fmt='d')
    plt.show()

    # Value count of test data
    print("Here is the values count of test data...")
    unique, counts = np.unique(y_test, return_counts=True)
    print(dict(zip(unique, counts)))


def make_corr_plot(pd_df, categorical_columns):
    f = plt.figure(figsize=(15, 10))

    df = pd_df.drop(categorical_columns, axis=1)
    df = df.fillna(df.median())

    # remove zero variance
    to_drop = pd.Series(df.columns.to_list(), index=df.columns.to_list())[df.std() == 0.0].to_list()
    df = df.drop(to_drop, axis=1)

    corr_mat = df.corr()
    plt.matshow(corr_mat, fignum=f.number)

    plt.xticks(range(df.shape[1]), df.columns, fontsize=12, rotation='vertical')
    plt.yticks(range(df.shape[1]), df.columns, fontsize=12)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    plt.title('Correlation Matrix', fontsize=16)


def corelation_output(drop_list, upper):
    for drop in drop_list:
        print('High corelation with {} is..:'.format(drop))
        print(upper[drop].sort_values(ascending=False)[upper[drop].sort_values(ascending=False) > 0.9])
        print('\n')


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return np.array([1 if y >= t else 0 for y in y_scores])


def precision_recall_threshold(p, r, thresholds, y_prob, y_true, Y_test, ts=[0.5]):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """

    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    for t in ts:
        print(f"Results for threshold is ..: {t}")
        labels = np.unique(y_true)
        y_pred_adj = adjusted_classes(y_prob, t)
        # confusion matrix
        data = confusion_matrix(Y_test, y_pred_adj)

        # calculate precision, recall, and F1 score
        accuracy = float(np.trace(data)) / np.sum(data)
        precision = precision_score(Y_test, y_pred_adj, average=None, labels=labels)[1]
        recall = recall_score(Y_test, y_pred_adj, average=None, labels=labels)[1]
        f1 = 2 * precision * recall / (precision + recall)
        print("accuracy:", accuracy)
        print("precision:", precision)
        print("recall:", recall)
        print("f1 score:", f1)

        df_cm = pd.DataFrame(confusion_matrix(Y_test, y_pred_adj),
                             columns=['No Default', 'Default'],
                             index=['No Default', 'Default'])
        df_cm.index.name = 'True Label'
        df_cm.columns.name = 'Predicted Label'
        plt.figure(figsize=(7, 5))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm, cmap='viridis', annot=True, fmt='d')
        plt.show()

        # Value count of test data
        print("0: Negative Class ~ Clean | 1: Positive Class ~ Malware")
        unique, counts = np.unique(Y_test, return_counts=True)
        print(dict(zip(unique, counts)))
        print("\n")
