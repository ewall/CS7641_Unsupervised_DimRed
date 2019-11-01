# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.ticker import MaxNLocator
from sklearn import random_projection
from yellowbrick.features import ParallelCoordinates
from util import *

SEED = 1
PLOT_DIR = "plots"
PKL_DIR = "pickles"


def optimize_components(X, feature_names, label, abbrev, chosen_n_components):
	# model selection: choose optimal number of components by reconstruction error
	n_components = np.arange(1, len(feature_names) + 1)
	rp_scores = []
	for n in n_components:
		rp = random_projection.GaussianRandomProjection(n_components=n, random_state=SEED)
		reduced = rp.fit_transform(X)
		rp_scores.append(get_reconstruction_error(X, reduced, rp))

	print(label + ": n_components with lowest RP reconstruction error = %d" % n_components[np.argmin(rp_scores)])
	print(label + ": chosen n_components by RP reconstruction error = %d" % chosen_n_components)
	print(label + ": chosen n_components' reconstruction error = " + str(rp_scores[chosen_n_components]))

	# create plot
	plt.figure()
	plt.plot(n_components, rp_scores, 'b', label='RP reconstruction error')
	plt.axvline(chosen_n_components, color='b',
	            label='RP components: %d' % chosen_n_components, linestyle='--')

	# format plot
	ax = plt.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.xlabel('number of components')
	plt.ylabel('reconstruction error')
	plt.legend(loc='lower right')
	plt.title(label + ": RP model selection")
	plt.savefig(path.join(PLOT_DIR, abbrev + "_rp_components.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	return chosen_n_components


if __name__ == "__main__":

	runs = (("data/creditcards_train.arff", "Credit Default", "d1", 13),
	        ("data/htru_train.arff", "Pulsar Detection", "d2", 6))

	for (fname, label, abbrev, n_components) in runs:
		X, y, feature_names = load_data(fname)

		# plot model selection (optimal number of components)
		optimize_components(X, feature_names, label, abbrev, n_components)

		# save as new set of features
		rp = random_projection.GaussianRandomProjection(n_components=n_components, random_state=SEED)
		start_time = time.perf_counter()
		df = pd.DataFrame(rp.fit_transform(X))
		run_time = time.perf_counter() - start_time
		print(label + ": run time = " + str(run_time))
		df.to_pickle(path.join(PKL_DIR, abbrev + "_rp.pickle"))

		# parallel coordinates plot
		visualizer = ParallelCoordinates(sample=0.2, shuffle=True, fast=True)
		visualizer.fit_transform(df, y)
		visualizer.ax.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45, horizontalalignment='right')
		visualizer.finalize()
		plt.savefig(path.join(PLOT_DIR, abbrev + "_rp_parallel.png"), bbox_inches='tight')
		visualizer.show()
		plt.close()

		#TODO why doesn't get_reconstruction_error(X, df, rp) work here?

