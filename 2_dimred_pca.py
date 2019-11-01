# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from yellowbrick.features import ParallelCoordinates
from util import *


SEED = 1
PLOT_DIR = "plots"
PKL_DIR = "pickles"
EV_THRESHOLD = 0.98


def optimize_components(X, feature_names, label, abbrev):
	# model selection (optimal number of components)
	# from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html

	def shrunk_cov_score(X):
		shrinkages = np.logspace(-2, 0, 30)
		cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages}, cv=5)
		return np.mean(cross_val_score(cv.fit(X).best_estimator_, X, cv=5))

	def lw_score(X):
		return np.mean(cross_val_score(LedoitWolf(), X, cv=5))

	# choose number of components by cross-validation
	n_components = np.arange(1, len(feature_names) + 1)
	pca = PCA(svd_solver='full', random_state=SEED)
	pca_scores = []
	for n in n_components:
		pca.n_components = n
		pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
	n_components_pca = n_components[np.argmax(pca_scores)]  #TODO change this formula???

	# choose number of components by Minka's MLE
	pca = PCA(svd_solver='full', n_components='mle', random_state=SEED)
	pca.fit(X)
	n_components_pca_mle = pca.n_components_

	# choose best n_components by explained variance >= EV_THRESHOLD
	ev = np.cumsum(pca.explained_variance_ratio_)
	chosen_n_components = np.argmax(ev >= EV_THRESHOLD) + 2

	# choose number of components with randomized SVD
	pca = PCA(svd_solver='randomized', random_state=SEED)
	pca_scores = []
	for n in n_components:
		pca.n_components = n
		pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
	n_components_random = n_components[np.argmax(pca_scores)]

	print(label + ": best n_components by PCA CV = %d" % n_components_pca)
	print(label + ": best n_components by PCA MLE = %d" % n_components_pca_mle)
	print(label + ": best n_components by PCA randomized SVD = %d" % n_components_random)
	print(label + ": chosen n_components by EV > 0.98 = %d" % chosen_n_components)

	# create plot
	plt.figure()
	plt.plot(n_components, pca_scores, 'b', label='PCA scores')
	plt.axvline(n_components_pca, color='b',
	            label='PCA CV: %d' % n_components_pca, linestyle='--')
	plt.axvline(n_components_pca_mle, color='k',
	            label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')
	plt.axvline(n_components_random, color='grey',
	            label='PCA random: %d' % n_components_random, linestyle='--')

	# compare with other covariance estimators
	plt.axhline(shrunk_cov_score(X), color='violet',
	            label='Shrunk Covariance MLE', linestyle='-.')
	plt.axhline(lw_score(X), color='orange',
	            label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

	# format plot
	ax = plt.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.xlabel('number of components')
	plt.ylabel('CV scores')
	plt.legend(loc='lower right')
	plt.title(label + ": PCA model selection")
	plt.savefig(path.join(PLOT_DIR, abbrev + "_pca_components.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	return chosen_n_components


if __name__ == "__main__":

	runs = (("data/creditcards_train.arff", "Credit Default", "d1"),
	        ("data/htru_train.arff", "Pulsar Detection", "d2"))

	for (fname, label, abbrev) in runs:
		X, y, feature_names = load_data(fname)

		# model selection (optimal number of components)
		n_components = optimize_components(X, feature_names, label, abbrev)

		# plot explained variance
		pca = PCA(random_state=SEED).fit(X)
		plt.figure()
		plt.plot(np.cumsum(pca.explained_variance_ratio_))
		plt.axvline(n_components, color='b', label='chosen number of components: %d' % n_components, linestyle='--')
		plt.xlabel('number of components')
		plt.ylabel('variance (%)')
		plt.title(label + ": Explained Variance by Number of Components")
		plt.savefig(path.join(PLOT_DIR, abbrev + "_pca_variance.png"), bbox_inches='tight')
		plt.show()
		plt.close()

		# save as new set of features
		pca = PCA(n_components=n_components, svd_solver='full', random_state=SEED)
		start_time = time.perf_counter()
		df = pd.DataFrame(pca.fit_transform(X))
		run_time = time.perf_counter() - start_time
		print(label + ": run time = " + str(run_time))
		df.to_pickle(path.join(PKL_DIR, abbrev + "_pca.pickle"))

		#TODO why doesn't this work?
		# output reconstruction error
		# recon_err = get_reconstruction_error(X, df, pca)
		# print(label + ": reconstruction error = " + str(recon_err))

		# parallel coordinates plot
		visualizer = ParallelCoordinates(sample=0.2, shuffle=True, fast=True)
		visualizer.fit_transform(df, y)
		visualizer.ax.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45, horizontalalignment='right')
		visualizer.finalize()
		plt.savefig(path.join(PLOT_DIR, abbrev + "_pca_parallel.png"), bbox_inches='tight')
		visualizer.show()
		plt.close()
