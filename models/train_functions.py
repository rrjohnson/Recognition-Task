import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def train_classifiers(X_train, y_train, X_test, y_test):
    """
    Train a number of classifiers on the training data
    """

    classifiers = {
        "Logistic Regression": LogisticRegression(penalty='l2'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(C=1.0, kernel='rbf'),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "ADA Boost": AdaBoostClassifier(),
        "GBC": GradientBoostingClassifier()
    }

    print('Training accuracies \n' + '-'*10)

    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train).mean()
        test_score = cross_val_score(classifier, X_test, y_test).mean()
        print('{} Training accuracy: {:.2f}%'.format(name,
                                                     training_score*100))

    print('\n' + '-'*60 + '\n')

    print('Test accuracies \n' + '-'*10)
    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        test_score = cross_val_score(classifier, X_test, y_test).mean()
        print('{} test accuracy: {:.2f}%'.format(name, test_score*100))

    return classifiers


def perform_gridsearch(classifiers, X_train, y_train):
    """ Perform a simple gridsearch for best parameters """

    # Define paramater ranges
    lr_params = {'penalty': ('l1', 'l2'), 'C': list(np.linspace(0.1, 1, 10))}

    knn_params = {'n_neighbors': [2, 3, 4], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    svm_params = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.01, 0.1],
                  'gamma': ['auto'], 'kernel': ['linear', 'rbf']}

    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(1,4,1)),
                   "min_samples_leaf": list(range(1,7,1))}

    forest_params = {"max_depth": [8, 9, 10, 11, 12],
                     "max_features": [1, 2],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 2, 3],
                     "bootstrap": [True, False],
                     "criterion": ["gini", "entropy"]}

    ada_params = {"n_estimators": [16, 17, 18],
                  "learning_rate": [0.02, 0.03, 0.04]
                  }

    gbc_params = {"n_estimators": range(100, 160, 3),
                  "learning_rate": [0.6],
                  "subsample": [0.4]
                  }


    parameter_grids = {'Logistic Regression': lr_params,
                       'KNN': knn_params,
                       'SVM': svm_params,
                       'Decision Tree': tree_params,
                       'Random Forest': forest_params,
                       'ADA Boost': ada_params,
                       'GBC': gbc_params}


    # Perform a grid search
    for name, classifier in classifiers.items():

        gs_clf = GridSearchCV(classifier, parameter_grids[name])
        gs_clf.fit(X_train, y_train)

        print('{} best parameters {}:'.format(name, gs_clf.best_params_))

        classifiers[name] = gs_clf.best_estimator_

    return classifiers


def calculate_cv(classifiers, X_train, y_train):

    for name, gs_clf in classifiers.items():
        score = cross_val_score(gs_clf, X_train, y_train, cv=5)

        print('{} cross validation score: {}'.format(name,
                                                     round(score.mean() * 100, 2).astype(str) + '%'))

