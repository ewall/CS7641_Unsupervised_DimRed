# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from yellowbrick.classifier import ConfusionMatrix
from util import *


ABBREV = "nn_kmeans"
LABEL = "Neural Net on K-Means"
SEED = 1
PLOT_DIR = "plots"


def get_features(model, data):
	part_1 = model.transform(data)  # cluster-distance space
	labels = model.predict(data)  # cluster labels
	labels = labels.reshape(-1, 1)
	enc = OneHotEncoder(categories='auto', sparse=False)
	enc.fit(labels)
	part_2 = enc.transform(labels)
	output = np.hstack((part_1, part_2))
	return pd.DataFrame(output)


if __name__ == "__main__":
	X_train, y_train, _ = load_data("data/htru_train.arff")
	X_test, y_test, _ = load_data("data/htru_test.arff")

	# make a feature set out of clusters
	model = KMeans(n_clusters=4, precompute_distances=True, random_state=SEED, n_jobs=-1)
	model.fit(X_train)
	X_train = get_features(model, X_train)
	X_test =  get_features(model, X_test)

	# build NN classifier
	clf = MLPClassifier(hidden_layer_sizes=(5),
	                    activation='logistic',
	                    solver='sgd',
	                    learning_rate_init=0.25,
	                    max_iter=1000,
	                    random_state=SEED,
	                    tol=0.000001,
	                    momentum=0.1)

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
