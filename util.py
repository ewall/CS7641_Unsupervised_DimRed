# Project 3: Unsupervised Learning & Dimension Reduction -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler


def load_data(filename):
	data = arff.loadarff(filename)
	dframe = pd.DataFrame(data[0])
	classes = dframe.pop('class')  # y is the column named 'class'
	classes = classes.astype(int)  # convert from binary/bytes to integers {0, 1}
	features = dframe.columns.values
	return dframe, classes, features


def load_xformed_data(origfile, reducedfile):
	origdata = arff.loadarff(origfile)
	classes = pd.DataFrame(origdata[0]).pop('class')  # y is the column named 'class'
	classes = classes.astype(int)  # convert from binary/bytes to integers {0, 1}
	dframe = pd.read_pickle(reducedfile)
	features = dframe.columns.values
	return dframe, classes, features


def rescale_data(X):
	# original data was z-scored and centered around zero, so just moving it to the non-negative range 0.0 to 1.0
	scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
	return scaler.fit_transform(X)


def get_reconstruction_error(X, reduced, model):
	projected = reduced.dot(model.components_)
	loss = ((X - projected) ** 2).mean().sum()
	return loss


def get_reconstruction_error_invertable(X, reduced, model):
	projected = model.inverse_transform(reduced)
	loss = ((X - projected) ** 2).mean().sum()
	return loss


if __name__ == "__main__":
	pass
