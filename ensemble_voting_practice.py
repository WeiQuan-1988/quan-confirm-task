from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB


def main():
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    estimators = [
        ('svm', SVC(gamma='scale', probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('logit', LogisticRegression(solver='lbfgs', max_iter=1000)),
        ('knn', KNeighborsClassifier()),
        ('nb',GaussianNB())
    ]

    accs = defaultdict(list)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in tqdm(list(skf.split(X,y))):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        voting = VotingClassifier(estimators)
        voting.fit(X_train, y_train)

        y_pred = voting.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs['voting'].append(acc)

        for name, estimator in voting.named_estimators_.items():
            y_pred = estimator.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accs[name].append(acc)
        
    for name, acc_list in accs.items():
        mean_acc = np.array(acc_list).mean()
        print(name,':',mean_acc)

if __name__ == '__main__':
    main()
