# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
from pandas.plotting import parallel_coordinates
from scipy.io import arff

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


runs = (("data/creditcards_std.arff", "Credit Default", "d1"),
        ("data/htru_std.arff", "Pulsar Detection", "d2"))

for (fname, label, abbrev) in runs:
	print("# Clustering " + label)
	X, y, feature_names = load_data(fname)

	# predict clusters
	y_pred = cluster.KMeans(n_clusters=2, precompute_distances=True, random_state=SEED, n_jobs=-1).fit_predict(X)

	# prep & save dataframe for plotting
	df = X.assign(cluster=y_pred)
	df.to_pickle(path.join(PKL_DIR, abbrev + "_kmeans_k2.pickle"))

	# Seaborn pairplot
	print("# Scatterplot for " + label)
	sns.set(style="ticks")
	grid = sns.pairplot(df, hue="cluster", vars=feature_names)
	plt.subplots_adjust(top=0.96)
	grid.fig.suptitle(label + ": K-means k=2")
	plt.savefig(path.join(PLOT_DIR, abbrev + "_kmeans_k2_scatter.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	# Pandas parallel coordinates plot
	print("# Parallel coordinates plot for " + label)
	ax = parallel_coordinates(df, "cluster", color=sns.color_palette().as_hex())
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
	plt.title(label + ": K-means k=2")
	plt.savefig(path.join(PLOT_DIR, abbrev + "_kmeans_k2_parallel.png"), bbox_inches='tight')
	plt.show()
	plt.close()
