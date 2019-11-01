# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import NMF
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.features import ParallelCoordinates
from util import *


SEED = 1
PLOT_DIR = "plots"
PKL_DIR = "pickles"


def rescale_data(X):
	# original data was z-scored and centered around zero, so just moving it to the non-negative range 0.0 to 1.0
	scaler = MinMaxScaler(feature_range=(0,1), copy=False)
	return scaler.fit_transform(X)


def optimize_components(X, feature_names, label, abbrev):
	# model selection (optimal number of components)
	# from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html

	# prepare explained variance scorer
	def get_score(model, data, scorer=explained_variance_score):
		prediction = model.inverse_transform(model.transform(data))
		return scorer(data, prediction)

	# choose number of components by explained variance
	n_components = np.arange(1, len(feature_names) + 1)
	nmf = NMF(random_state=SEED)
	nmf_scores = []
	for n in n_components:
		nmf.n_components = n
		nmf.fit(X)
		nmf_scores.append(get_score(nmf, X))
	nmf_scores = np.array(nmf_scores)
	n_components_nmf = n_components[np.argmax(nmf_scores >= 0.95)]
	print(label + ": best n_components by explained variance > 0.95 = %d" % int(n_components_nmf))

	# create plot
	plt.figure()
	plt.plot(n_components, nmf_scores, 'b', label='explained variance by num/components')
	plt.axvline(n_components_nmf, color='b',
	            label='chosen number of components: %d' % n_components_nmf, linestyle='--')

	# format plot
	ax = plt.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.xlabel('number of components')
	plt.ylabel('explained variance')
	plt.legend(loc='lower right')
	plt.title(label + ": NMF model selection")
	plt.savefig(path.join(PLOT_DIR, abbrev + "_nmf_components.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	return n_components_nmf


if __name__ == "__main__":

	runs = (("data/creditcards_train.arff", "Credit Default", "d1"),
	        ("data/htru_train.arff", "Pulsar Detection", "d2"))

	for (fname, label, abbrev) in runs:
		X, y, feature_names = load_data(fname)
		X = rescale_data(X)  # data was z-scored and centered around zero, so needs to be moved to non-negative

		# model selection (optimal number of components)
		n_components = optimize_components(X, feature_names, label, abbrev)

		# save as new set of features
		nmf = NMF(n_components=n_components, random_state=SEED)
		start_time = time.perf_counter()
		df = pd.DataFrame(nmf.fit_transform(X))
		run_time = time.perf_counter() - start_time
		print(label + ": run time = " + str(run_time))
		df.to_pickle(path.join(PKL_DIR, abbrev + "_nmf.pickle"))

		# parallel coordinates plot
		visualizer = ParallelCoordinates(sample=0.2, shuffle=True, fast=True)
		visualizer.fit_transform(df, y)
		visualizer.ax.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45, horizontalalignment='right')
		visualizer.finalize()
		plt.savefig(path.join(PLOT_DIR, abbrev + "_nmf_parallel.png"), bbox_inches='tight')
		visualizer.show()
		plt.close()

		# output reconstruction error
		print(label + ": reconstruction error = " + str(nmf.reconstruction_err_))
