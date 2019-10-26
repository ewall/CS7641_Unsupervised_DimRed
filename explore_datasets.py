# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import arff
from yellowbrick.features.pca import PCADecomposition
from yellowbrick.target import FeatureCorrelation

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


runs = (("data/creditcards_std.arff", "Credit Default", "d1", 6),
        ("data/htru_std.arff", "Pulsar Detection", "d2", 4))

for (fname, label, abbrev, best_k) in runs:
	X, y, feature_names = load_data(fname)

	# Pearson correlation with class
	visualizer = FeatureCorrelation(method='pearson', labels=feature_names)
	visualizer.fit(X, y)
	visualizer.finalize()
	plt.savefig(path.join(PLOT_DIR, abbrev + "_expore_correl.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()

	# mutual information with class
	visualizer = FeatureCorrelation(method='mutual_info-classification')
	visualizer.fit(X, y)
	visualizer.finalize()
	plt.savefig(path.join(PLOT_DIR, abbrev + "_expore_mutual.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()

	# Pearson correlation with other features
	feature_corr = X.corr()
	mask = np.zeros_like(feature_corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True
	plt.figure(figsize=(5, 5))
	sns.heatmap(feature_corr,
	            vmin=-1,
	            cmap='coolwarm',
	            mask=mask)
	ax = plt.gca()
	ax.set_yticklabels(ax.get_yticklabels(), rotation=45, verticalalignment='top')
	plt.savefig(path.join(PLOT_DIR, abbrev + "_expore_heatmap.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	# PCA Projection
	colors = np.array(['r' if yi else 'b' for yi in y])
	visualizer = PCADecomposition(scale=True, color=colors)
	visualizer.fit_transform(X, y)
	visualizer.finalize()
	plt.savefig(path.join(PLOT_DIR, abbrev + "_expore_pca.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()

	# PCA Projection Biplot
	visualizer = PCADecomposition(scale=True, proj_features=True)
	visualizer.fit_transform(X, y)
	visualizer.finalize()
	plt.savefig(path.join(PLOT_DIR, abbrev + "_expore_biplot.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()
