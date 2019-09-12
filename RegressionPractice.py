#Practice of linear regression on Iris dataset

from sklearn import datasets, linear_model
# from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target
print("X.shape:{}".format(X.shape))
print("y.shape:{}".format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print("X_train.shape:{}".format(X_train.shape))
print("X_test.shape:{}".format(X_test.shape))
print("y_train.shape:{}".format(y_train.shape))
print("y_tst.shape:{}".format(y_test.shape))

LogisticRegression = linear_model.LogisticRegression(solver='liblinear')
LogisticRegression.fit(X_train, y_train)
print('Logistic Regression Coefficients:%s,intercept%s'%(LogisticRegression.coef_, LogisticRegression.intercept_))
print("Logistic Regression score on test:%.2f"%LogisticRegression.score(X_test, y_test))

LinearRegression = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
LinearRegression.fit(X_train, y_train)
print('Linear Regression Coefficients:%s,intercept%s'%(LinearRegression.coef_, LinearRegression.intercept_))
print("Linear Regression score on test:%.2f"%LinearRegression.score(X_test, y_test))
