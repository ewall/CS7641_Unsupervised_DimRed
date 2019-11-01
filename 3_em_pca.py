# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import itertools
import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn import metrics
from sklearn import mixture
from yellowbrick.features import ParallelCoordinates
from util import *

SEED = 1
PLOT_DIR = "plots"
PKL_DIR = "pickles"


runs = (("data/creditcards_train.arff", "Credit Default", "d1", 2, 'spherical'),
        ("data/htru_train.arff", "Pulsar Detection", "d2", 2, 'full'))

for (fname, label, abbrev, best_k, best_type) in runs:
	X, y, feature_names = load_xformed_data(fname, path.join(PKL_DIR, abbrev + "_pca.pickle"))

	# find optimal k with BIC, Calinski-Harabasz, & Silhouette scores
	# from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
	lowest_bic = np.infty
	bic = []
	ch = []
	silho = []

	if len(feature_names) > 10:
		n_components_range = range(2, len(feature_names) + 1)
	else:
		n_components_range = range(2, 2 * len(feature_names) + 1)  # try bigger range for the small-d dataset

	cv_types = ['spherical', 'tied', 'diag', 'full']
	for cv_type in cv_types:
		for n_components in n_components_range:
			print("# Optimizing k for " + label + " with k=" + str(n_components) + ", " + cv_type)
			# get a Gaussian mixture with EM
			gmm = mixture.GaussianMixture(n_components=n_components,
			                              covariance_type=cv_type,
			                              n_init=3,
			                              init_params='random',
			                              random_state=SEED)
			y_pred = gmm.fit_predict(X)
			# calc BIC score
			bic.append(gmm.bic(X))
			if bic[-1] < lowest_bic:
				lowest_bic = bic[-1]
				best_gmm = gmm
			# calc CH score
			ch.append(metrics.calinski_harabasz_score(X, y_pred))
			# calc Silhouette
			silho.append(metrics.silhouette_score(X, y_pred, metric='euclidean'))
	bic = np.array(bic)
	ch = np.array(ch)
	silho = np.array(silho)

	# plot the BIC scores
	print("# Plotting BIC for " + label)
	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
	bars = []
	plt.figure(figsize=(8, 6))
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, bic[i * len(n_components_range):
		                              (i + 1) * len(n_components_range)],
		                    width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	plt.xlabel('Number of components')
	plt.legend([b[0] for b in bars], cv_types)
	plt.title(label + ": BIC scores for Expectation-Maximization")
	plt.subplots_adjust(hspace=.35, bottom=.02)
	plt.savefig(path.join(PLOT_DIR, abbrev + "_em-pca_bic.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	# plot the CH scores
	print("# Plotting Calinski-Harabasz for " + label)
	bars = []
	plt.figure(figsize=(8, 6))
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, ch[i * len(n_components_range):
		                              (i + 1) * len(n_components_range)],
		                    width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([ch.min() * 1.01 - .01 * ch.max(), ch.max()])
	plt.title('Calinski-Harabasz score per model')
	plt.xlabel('Number of components')
	plt.legend([b[0] for b in bars], cv_types)
	plt.title(label + ": Calinski-Harabasz scores for EM")
	plt.subplots_adjust(hspace=.35, bottom=.02)
	plt.savefig(path.join(PLOT_DIR, abbrev + "_em-pca_ch.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	# plot the silhouette scores
	print("# Plotting Silhouette for " + label)
	bars = []
	plt.figure(figsize=(8, 6))
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, silho[i * len(n_components_range):
		                              (i + 1) * len(n_components_range)],
		                    width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([silho.min() * 1.01 - .01 * silho.max(), silho.max()])
	plt.title('Silhouette score per model')
	plt.xlabel('Number of components')
	plt.legend([b[0] for b in bars], cv_types)
	plt.title(label + ": Silhouette Coefficients for EM")
	plt.subplots_adjust(hspace=.35, bottom=.02)
	plt.savefig(path.join(PLOT_DIR, abbrev + "_em-pca_silhouette.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	# predict best clusters
	model = mixture.GaussianMixture(n_components=best_k, covariance_type=best_type, random_state=SEED)
	start_time = time.perf_counter()
	y_pred = model.fit_predict(X)
	run_time = time.perf_counter() - start_time
	print(label + ": run time = " + str(run_time))
	print(label + ": iterations until convergence = " + str(model.n_iter_))
	df = X.assign(cluster=y_pred)
	df.to_pickle(path.join(PKL_DIR, abbrev + "_em-pca.pickle"))

	# pairplot
	print("# Scatterplot for " + label)
	sns.set(style="ticks")
	grid = sns.pairplot(df, hue="cluster", vars=feature_names)
	plt.subplots_adjust(top=0.96)
	grid.fig.suptitle(label + ": K-means k=" + str(best_k))
	plt.savefig(path.join(PLOT_DIR, abbrev + "_em-pca_scatter.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	# parallel coordinates plot
	print("# Parallel Coordinates Plot for " + label)
	visualizer = ParallelCoordinates(features=feature_names, sample=0.1, shuffle=True, fast=True)
	visualizer.fit_transform(X, y_pred)
	visualizer.ax.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45, horizontalalignment='right')
	visualizer.finalize()
	plt.savefig(path.join(PLOT_DIR, abbrev + "_em-pca_parallel.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()

	# compare with ground truth (classes)
	print(label + ": Homogeneity Score = " + str(metrics.homogeneity_score(y, y_pred)))
	print(label + ": V Measure Score = " + str(metrics.v_measure_score(y, y_pred)))
	print(label + ": Mutual Info Score = " + str(metrics.mutual_info_score(y, y_pred)))
	print(label + ": Adjusted Rand Index = " + str(metrics.adjusted_rand_score(y, y_pred)))
