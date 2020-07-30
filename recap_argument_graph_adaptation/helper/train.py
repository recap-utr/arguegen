from sklearn import svm

svc = svm.LinearSVC()
svc.fit(X_train, y_train)
