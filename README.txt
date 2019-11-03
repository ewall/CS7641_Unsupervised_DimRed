Project 3: Unsupervised Learning & Dimensionality Reduction
###########################################################
GT CS7641 Machine Learning, Fall 2019
Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

## Background ##
Classwork for Georgia Tech's CS7641 Machine Learning course. Project code should be published publicly for grading purposes, under the assumption that students should not plagiarize content and must do their own analysis.

These experiments compare 2 clustering algorithms and 4 dimensionality reduction algorithms, and also compare training of a neural network on the output of each of the above.

## Requirements ##

* Python 3.7 or higher
* essential Python machine learning libraries such as Numpy, Pandas, and Scipy
* Python visualization libraries include Matplotlib, Seaborn, and Yellowbrick

## Instructions ##

* Clone this source repository from Github onto your computer using the following command:
	`git clone git@github.com:ewall/CS7641_Unsupervised_DimRed.git`

* From the source directory, run the following command to install the necessary Python modules:
	`pip install -r requirements.txt`

* Directories in the project include:
	* `data` contains the pre-processed datasets
	* `plots` is where the scripts will save *.png files of the plots
	* `pickles` is where the scripts save serialized dataframes after processing
	* `experiments` contains miscellaenous scripts which have only a passing mention in the analysis paper

* Here is a guide to the filenames of the Python scripts in the directory:
	* `explore_datasets.py` shows plots and statistics of the original datasets
	* `1_clust_*.py` are the clustering experiments for the 2 cluster alogrithms
	* `2_dimred_*.py` are the dimension-reduction experiments for the 4 algorithms
	* `3_em_*.py` are EM clustering of the reduced feature sets from the 4 algorithms; similarly, `3_kmeans_*.py` are K-Means clustering across the same 4 reduced sets
	* `4_nn_base.py` is the baseline neural network run on the original data, for comparison
	* `4_nn_dimred.py` is the neural network run for all 4 feature selection algorithms
	* `5_nn_*.py` are the neural network runs on the 2 clustering algorithms
