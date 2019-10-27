# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from matplotlib.ticker import MaxNLocator
from scipy.io import arff
from sklearn.decomposition import FastICA
from yellowbrick.features import ParallelCoordinates

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


def optimize_components(X, feature_names, label, abbrev):
	# model selection (optimal number of components)

	# choose number of components by highest average kurtosis
	n_components = np.arange(1, len(feature_names) + 1, 5)
	ica = FastICA(random_state=SEED)
	ica_scores = []
	for n in n_components:
		ica.n_components = n
		result = pd.DataFrame(ica.fit_transform(X))
		ica_scores.append(result.kurtosis().mean())
	n_components_ica = n_components[np.argmax(ica_scores)]
	print(label + ": best n_components by ICA CV = %d" % n_components_ica)

	# create plot
	plt.figure()
	plt.plot(n_components, ica_scores, 'b', label='PCA scores')
	plt.axvline(n_components_ica, color='b',
	            label='ICA CV: %d' % n_components_ica, linestyle='--')
	ax = plt.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.xlabel('number of components')
	plt.ylabel('average kurtosis')
	plt.legend(loc='lower right')
	plt.title(label + ": ICA model selection")
	plt.savefig(path.join(PLOT_DIR, abbrev + "_ica_components.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	return n_components_ica


runs = (("data/creditcards_train.arff", "Credit Default", "d1"),
        ("data/htru_train.arff", "Pulsar Detection", "d2"))

for (fname, label, abbrev) in runs:
	X, y, feature_names = load_data(fname)

	# model selection (optimal number of components)
	n_components = optimize_components(X, feature_names, label, abbrev)

	# save as new set of features
	ica = FastICA(n_components=n_components, random_state=SEED)
	start_time = time.perf_counter()
	df = pd.DataFrame(ica.fit_transform(X))
	run_time = time.perf_counter() - start_time
	print(label + ": run time = " + str(run_time))
	print(label + ": iterations until convergence = " + str(ica.n_iter_))
	df.to_pickle(path.join(PKL_DIR, abbrev + "_ica.pickle"))

	# parallel coordinates plot
	visualizer = ParallelCoordinates(sample=0.2, shuffle=True, fast=True)
	visualizer.fit_transform(df, y)
	visualizer.ax.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45, horizontalalignment='right')
	visualizer.finalize()
	plt.savefig(path.join(PLOT_DIR, abbrev + "_pca_parallel.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()
