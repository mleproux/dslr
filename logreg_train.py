import pandas as pd
import numpy as np
import math
import json


LEARNING_RATE = 0.01
ITERATIONS = 1000


houses = [
	"Gryffindor",
	"Hufflepuff",
	"Ravenclaw",
	"Slytherin"
]


def load(path: str) -> pd.DataFrame:
	"""Load a CSV file into a map object.
	"""
	try:
		dataset = pd.read_csv(path)
		return dataset
	except FileNotFoundError:
		print("Error: File not found.")
	except pd.errors.EmptyDataError:
		print("Error: File empty or invalid.")
	except pd.errors.ParserError:
		print("Error: File format invalid.")
	except Exception as e:
		print(f"Error: An unexpected error occurred: {e}")
	return None


def fill_nan_with_mean(dataset):
	"""Replace NaN values with the mean of each column.
	"""
	data = np.array(dataset)
	for col in range(13):
		sum = 0.0
		count = 0
		for i in range(len(data)):
			if not math.isnan(data[i][col + 6]):
				sum += data[i][col + 6]
				count += 1
		mean = sum / count
		for i in range(len(data)):
			if math.isnan(data[i][col + 6]):
				data[i][col + 6] = mean
	return data


def standardize(X):
	"""Standardize the dataset by centering and scaling the features.
	"""
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X_scaled = (X - mean) / std
	return X_scaled


def get_values(data):
	"""Extract the numeric values from the data and standardize them.
	"""
	values = [[0.0 for _ in range(13)] for _ in range(len(data))]
	for i in range(len(data)):
		for col in range(13):
			values[i][col] = float(data[i][col + 6])
	values = standardize(values)
	return values


def sigmoid(z) -> float:
	"""Compute the sigmoid function.
	"""
	return 1 / (1 + np.exp(-z))


def gradient_descent(X, y, lr=LEARNING_RATE, iter=ITERATIONS):
	"""Train a binary logistic regression classifier using gradient descent.
	"""
	theta = np.zeros(13)
	for _ in range(iter):
		h = sigmoid(X @ theta)
		gradient = (1 / len(X)) * (np.transpose(X) @ (h - y))
		theta -= lr * gradient
	return theta


def get_weights(dataset, values):
	"""Train a binary logistic regression classifier for each house and return the weights.
	"""
	weights = {}
	for house in houses:
		y = dataset["Hogwarts House"]
		y_binary = (y == house).astype(int)
		theta = gradient_descent(np.array(values), y_binary)
		weights[house] = theta.tolist()
	return weights


def save_weights(weights):
	"""Save the weights in a JSON file.
	"""
	with open("weights.json", "w") as f:
		json.dump(weights, f)


def main():
	dataset = load('datasets/dataset_train.csv')
	if dataset is None:
		exit(1)

	data = fill_nan_with_mean(dataset)
	values = get_values(data)
	weights = get_weights(dataset, values)
	save_weights(weights)


if __name__ == "__main__":
	main()
