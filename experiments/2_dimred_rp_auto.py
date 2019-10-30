# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import pandas as pd
from scipy.io import arff
from sklearn import random_projection

SEED = 1
PLOT_DIR = "plots"
PKL_DIR = "pickles"


def load_data(filename):
	data = arff.loadarff(filename)
	dframe = pd.DataFrame(data[0])
	classes = dframe.pop('class')  # y is the column named 'class'
	classes = classes.astype(int)  # convert from binary/bytes to integers {0, 1}
	features = dframe.columns.values
	return dframe, classes, features


if __name__ == "__main__":

	runs = (("../data/creditcards_train.arff", "Credit Default", "d1", 13),
	        ("../data/htru_train.arff", "Pulsar Detection", "d2", 6))

	for (fname, label, abbrev, n_components) in runs:
		X, y, feature_names = load_data(fname)

		for eps in (0.1, 0.5, 0.999999):

			try:
				rp = random_projection.GaussianRandomProjection(n_components='auto', eps=eps, random_state=SEED).fit(X)
			except Exception as e:
				print(label + ": " + str(e))
				pass
