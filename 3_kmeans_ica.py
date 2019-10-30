# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import os.path as path
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.ticker import MaxNLocator
from sklearn import cluster, metrics
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.features import ParallelCoordinates
from util import *

SEED = 1
PLOT_DIR = "plots"
PKL_DIR = "pickles"


if __name__ == "__main__":

	runs = (("data/creditcards_train.arff", "Credit Default", "d1", 7),
	        ("data/htru_train.arff", "Pulsar Detection", "d2", 6))

	for (fname, label, abbrev, best_k) in runs:
		X, y, feature_names = load_xformed_data(fname, path.join(PKL_DIR, abbrev + "_ica.pickle"))

		if len(feature_names) > 10:
			k_range = range(2, len(feature_names) + 1)
		else:
			k_range = range(2, 2 * len(feature_names) + 1)  # try bigger range for the small-d dataset

		# find optimal k with elbow method
		for metric in ('distortion', 'silhouette', 'calinski_harabasz'):
			print("# Optimizing k for " + label + " with " + metric)
			model = cluster.KMeans(precompute_distances=True, random_state=SEED, n_jobs=-1)
			try:
				visualizer = KElbowVisualizer(model, k=k_range, metric=metric, locate_elbow=True)
			except:
				visualizer = KElbowVisualizer(model, k=k_range, metric=metric, locate_elbow=False)
			visualizer.fit(X)
			visualizer.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
			visualizer.finalize()
			plt.savefig(path.join(PLOT_DIR, abbrev + "_km-ica_elbow_" + metric + ".png"), bbox_inches='tight')
			visualizer.show()
			plt.close()

		# predict best clusters
		print("# Clustering " + label)
		model = cluster.KMeans(n_clusters=best_k, precompute_distances=True, random_state=SEED, n_jobs=-1)
		start_time = time.perf_counter()
		y_pred = model.fit_predict(X)
		run_time = time.perf_counter() - start_time
		print(label + ": run time = " + str(run_time))
		print(label + ": iterations until convergence = " + str(model.n_iter_))
		df = X.assign(cluster=y_pred)
		df.to_pickle(path.join(PKL_DIR, abbrev + "_km-ica.pickle"))

		# silhouette plot
		print("# Silhouette Visualizer for " + label)
		visualizer = SilhouetteVisualizer(model)
		visualizer.fit(X)
		visualizer.finalize()
		plt.savefig(path.join(PLOT_DIR, abbrev + "_km-ica_silhouette.png"), bbox_inches='tight')
		visualizer.show()
		plt.close()

		# pairplot
		print("# Scatterplot for " + label)
		sns.set(style="ticks")
		grid = sns.pairplot(df, hue="cluster", vars=feature_names)
		plt.subplots_adjust(top=0.96)
		grid.fig.suptitle(label + ": K-means k=" + str(best_k))
		plt.savefig(path.join(PLOT_DIR, abbrev + "_km-ica_scatter.png"), bbox_inches='tight')
		plt.show()
		plt.close()

		# parallel coordinates plot
		print("# Parallel Coordinates Plot for " + label)
		visualizer = ParallelCoordinates(features=feature_names, sample=0.1, shuffle=True, fast=True)
		visualizer.fit_transform(X, y_pred)
		visualizer.ax.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45, horizontalalignment='right')
		visualizer.finalize()
		plt.savefig(path.join(PLOT_DIR, abbrev + "_km-ica_parallel.png"), bbox_inches='tight')
		visualizer.show()
		plt.close()

		# compare with ground truth (classes)
		print(label + ": Homogeneity Score = " + str(metrics.homogeneity_score(y, y_pred)))
		print(label + ": V Measure Score = " + str(metrics.v_measure_score(y, y_pred)))
		print(label + ": Mutual Info Score = " + str(metrics.mutual_info_score(y, y_pred)))
		print(label + ": Adjusted Rand Index = " + str(metrics.adjusted_rand_score(y, y_pred)))
