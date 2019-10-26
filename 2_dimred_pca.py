# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.io import arff
from sklearn.decomposition import PCA
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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


def shrunk_cov_score(X):
	shrinkages = np.logspace(-2, 0, 30)
	cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages}, cv=5)
	return np.mean(cross_val_score(cv.fit(X).best_estimator_, X, cv=5))


def lw_score(X):
	return np.mean(cross_val_score(LedoitWolf(), X, cv=5))


runs = (("data/creditcards_train.arff", "Credit Default", "d1"),
        ("data/htru_train.arff", "Pulsar Detection", "d2"))
# runs = (("data/htru_train.arff", "Pulsar Detection", "d2"),)

for (fname, label, abbrev) in runs:
	X, y, feature_names = load_data(fname)

	# model selection (optimal number of components)
	# from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html

	# choose number of components by cross-validation
	n_components = np.arange(0, len(feature_names) + 1, 5)
	pca = PCA(svd_solver='full', random_state=SEED)
	pca_scores = []
	for n in n_components:
		pca.n_components = n
		pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
	n_components_pca = n_components[np.argmax(pca_scores)]

	# choose number of components by Minka's MLE
	pca = PCA(svd_solver='full', n_components='mle', random_state=SEED)
	pca.fit(X)
	n_components_pca_mle = pca.n_components_

	print(label + ": best n_components by PCA CV = %d" % n_components_pca)
	print(label + ": best n_components by PCA MLE = %d" % n_components_pca_mle)

	# create plot
	plt.figure()
	plt.plot(n_components, pca_scores, 'b', label='PCA scores')
	plt.axvline(n_components_pca, color='b',
	            label='PCA CV: %d' % n_components_pca, linestyle='--')
	plt.axvline(n_components_pca_mle, color='k',
	            label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

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
	plt.savefig(path.join(PLOT_DIR, abbrev + "_pca.png"), bbox_inches='tight')
	plt.show()
	plt.close()

