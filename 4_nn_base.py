# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import matplotlib.pyplot as plt
import os.path as path
import time
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from yellowbrick.classifier import ConfusionMatrix
from util import *

ABBREV = "nn_base"
LABEL = "Neural Net Baseline"
SEED = 1
PLOT_DIR = "plots"
PKL_DIR = "pickles"

if __name__ == "__main__":
	X_train, y_train, feature_names = load_data("data/htru_train.arff")
	X_test, y_test, _ = load_data("data/htru_test.arff")

	clf = MLPClassifier(hidden_layer_sizes=(5),
	                    activation='logistic',
	                    solver='sgd',
	                    learning_rate_init=0.25,
	                    max_iter=1000,
	                    random_state=SEED,
	                    tol=0.000001,
	                    momentum=0.1)
						# early_stopping=True,
						# validation_fraction=0.1,
						# n_iter_no_change=20,
						# verbose = True)

	# train
	start_time = time.perf_counter()
	clf.fit(X_train, y_train)
	training_time = time.perf_counter() - start_time
	print("training time = " + str(training_time))
	print("stopped at iterations = " + str(clf.n_iter_))

	# test on train set
	y_train_pred = clf.predict(X_train)
	train_kappa = metrics.cohen_kappa_score(y_train, y_train_pred)
	train_error = metrics.accuracy_score(y_train, y_train_pred)
	print("TRAINING kappa=" + str(train_kappa))
	print("TRAINING error=" + str(train_error))

	# test
	y_test_pred = clf.predict(X_test)
	test_kappa = metrics.cohen_kappa_score(y_test, y_test_pred)
	test_error = metrics.accuracy_score(y_test, y_test_pred)
	print("TEST kappa=" + str(test_kappa))
	print("TEST error=" + str(test_error))

	# plot confusion matrix
	title = LABEL + ": Confusion Matrix"
	cm = ConfusionMatrix(clf, classes=[0, 1], title=title)
	cm.fit(X_train, y_train)
	cm.score(X_test, y_test)
	cm.show(outpath=path.join(PLOT_DIR, ABBREV + "_confusion.png"))
	cm.show()
	plt.close()
