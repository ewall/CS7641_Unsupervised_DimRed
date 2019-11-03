# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import FastICA
from yellowbrick.features import ParallelCoordinates
from util import *


SEED = 1
PLOT_DIR = "plots"
PKL_DIR = "pickles"


def optimize_components(X, feature_names, label, abbrev):
	# model selection (optimal number of components)

	# choose number of components by highest average kurtosis
	n_components = np.arange(1, len(feature_names) + 1)
	ica = FastICA(random_state=SEED)
	ica_scores = []
	for n in n_components:
		ica.n_components = n
		result = pd.DataFrame(ica.fit_transform(X))
		ica_scores.append(result.kurtosis().mean())
	n_components_ica = n_components[np.argmax(ica_scores)]
	print(label + ": best n_components by average kurtosis = %d" % n_components_ica)

	# create plot
	plt.figure()
	plt.plot(n_components, ica_scores, 'b', label='average kurtosis by number of components')
	plt.axvline(n_components_ica, color='b',
	            label='chosen number of components: %d' % n_components_ica, linestyle='--')
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
	plt.savefig(path.join(PLOT_DIR, abbrev + "_ica_parallel.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()

	# explore kurtosis
	print(label + ": kurtosis by feature = ", df.kurtosis())
	print(label + ": mean kurtosis = ", df.kurtosis().mean())

	# output reconstruction error
	recon_err = get_reconstruction_error_invertable(X, df, ica)
	print(label + ": reconstruction error = " + str(recon_err))
