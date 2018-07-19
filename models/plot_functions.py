import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle


def learning(classifiers, X, y):
    """
    Plot the learning curves of an arbitrary number of classifiers
    """

    # Shuffle first to alleviate learning curve bug
    X, y = shuffle(X, y)

    # Generate plots
    num_rows = int(np.ceil(len(classifiers) / 2))
    fig, ax = plt.subplots(num_rows, 2, figsize=(22, 7*num_rows))
    ax = ax.ravel()


    for i, (name, classifier) in enumerate(classifiers.items()):

        train_sizes = np.linspace(0.1, 1, 5)
        train_sizes, train_scores, test_scores = learning_curve(classifier, X, y, train_sizes=train_sizes, cv=5)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        ax[i].plot(train_sizes, train_mean, marker='o', label='Training score')
        ax[i].plot(train_sizes, test_mean, marker='o', label='Cross validation score')

        ax[i].fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1,
                        color="#2492ff")
        ax[i].fill_between(train_sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.1,
                        color="#ff9124")

        ax[i].set_xlabel('Training sizes')
        ax[i].set_ylabel('Accuracy')
        ax[i].set_title(name, fontsize=14)
        ax[i].legend()


def roc(classifiers, X, y):

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for name, clf in classifiers.items():

        try:
            y_pred = clf.predict_proba(X)[::,1]
            fpr, tpr, _ = roc_curve(y, y_pred)
            auc = roc_auc_score(y, y_pred)
            legend = '{}: {:.3f}'.format(name, auc)
            ax.plot(fpr, tpr, label=legend)
            ax.save
        except:
            pass

    # Plot the 50:50 line
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '--', color='black', linewidth=1)

    ax.set_title('ROC')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend();

def roc_ollie(classifiers, X, y):

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Plot the x = y line
    plt.plot([0.0, 1.0], [0.0, 1.0], color='grey', lw=1.0, linestyle='--')
    plt.gca().fill_between([0.0, 1.0], [0.0, 1.0], where=[True] * 2, interpolate=True, color='lightgray')

    # Plot ROC for all classifiers
    for name, classifier in classifiers.items():
        if name != 'SVM':
            y_probs = classifier.predict_proba(X)[::,1]

            fpr, tpr, _ = roc_curve(y, y_probs)
            legend = '{}: {:.2f}'.format(name, roc_auc_score(y, y_probs))
            plt.plot(fpr, tpr, label=legend)

            # The True positive rates achieved, and the false positive rates achieved.
            #  cm = confusion_matrix(y, pred)
            #  tps, fps = 1.0 * cm[1, 1] / cm[1, :].sum(), 1.0 * cm[0, 1] / cm[0,:].sum()
            #  plt.plot(fps, tps, 'ko', ms=10, alpha=0.5)


    # Global plot options
    plt.xlim([0, 1])
    plt.ylim([0.0, 1])
    plt.title('ROC', y=1.02)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.tick_params(pad=10)
    plt.gca().set_aspect(1.0)

