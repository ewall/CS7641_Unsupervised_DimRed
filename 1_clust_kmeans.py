# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
from matplotlib.ticker import MaxNLocator
from scipy.io import arff
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
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


runs = (("data/creditcards_train.arff", "Credit Default", "d1", 6),
        ("data/htru_train.arff", "Pulsar Detection", "d2", 4))

for (fname, label, abbrev, best_k) in runs:
	X, y, feature_names = load_data(fname)

	if len(feature_names) > 10:
		k_range = range(2, len(feature_names) + 1)
	else:
		k_range = range(2, 2 * len(feature_names) + 1)  # try bigger range for the small-d dataset

	# find optimal k with elbow method
	for metric in ('distortion', 'silhouette', 'calinski_harabasz'):
		print("# Optimizing k for " + label + " with " + metric)
		model = cluster.KMeans(precompute_distances=True, random_state=SEED, n_jobs=-1)
		visualizer = KElbowVisualizer(model, k=k_range, metric=metric, locate_elbow=False)
		visualizer.fit(X)
		visualizer.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		visualizer.finalize()
		plt.savefig(path.join(PLOT_DIR, abbrev + "_kmeans_elbow_" + metric + ".png"), bbox_inches='tight')
		visualizer.show()
		plt.close()

	# predict best clusters
	print("# Clustering " + label)
	model = cluster.KMeans(n_clusters=best_k, precompute_distances=True, random_state=SEED, n_jobs=-1)
	y_pred = model.fit_predict(X)
	df = X.assign(cluster=y_pred)
	df.to_pickle(path.join(PKL_DIR, abbrev + "_kmeans.pickle"))  # save dataframe

	# silhouette plot
	print("# Silhouette Visualizer for " + label)
	visualizer = SilhouetteVisualizer(model)
	visualizer.fit(X)
	visualizer.finalize()
	plt.savefig(path.join(PLOT_DIR, abbrev + "_kmeans_silhouette.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()

	# pairplot
	print("# Scatterplot for " + label)
	sns.set(style="ticks")
	grid = sns.pairplot(df, hue="cluster", vars=feature_names)
	plt.subplots_adjust(top=0.96)
	grid.fig.suptitle(label + ": K-means k=" + str(best_k))
	plt.savefig(path.join(PLOT_DIR, abbrev + "_kmeans_scatter.png"), bbox_inches='tight')
	plt.show()
	plt.close()

	# parallel coordinates plot
	print("# Parallel Coordinates Plot for " + label)
	visualizer = ParallelCoordinates(features=feature_names, sample=0.1, shuffle=True, fast=True)
	visualizer.fit_transform(X, y_pred)
	visualizer.ax.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45, horizontalalignment='right')
	visualizer.finalize()
	plt.savefig(path.join(PLOT_DIR, abbrev + "_kmeans_parallel.png"), bbox_inches='tight')
	visualizer.show()
	plt.close()
