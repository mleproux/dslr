import pandas as pd
import numpy as np
import math
import json
import csv
# from sklearn.metrics import accuracy_score


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


def predict(values, weights):
	"""Predict the houses the students belongs to.
	"""
	predictions = []
	for i in range(len(values)):
		best_house = None
		best_score = -1
		for house, theta in weights.items():
			theta = np.array(theta)
			score = sigmoid(values[i] @ theta)
			if score > best_score:
				best_score = score
				best_house = house
		predictions.append(best_house)
	return predictions


def save_in_csv_file(predictions):
	"""Write the predictions to a CSV file.
	"""
	with open("houses.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["Index", "Hogwarts House"])
		for i, pred in enumerate(predictions):
			writer.writerow([i, pred])


def compare_predictions(predictions):
	"""Compare the predictions against the expected values
	using the accuracy score from the Scikit-Learn library
	and print the accuracy.
	"""
	dataset = load('datasets/dataset_train.csv')
	if dataset is None:
		exit(1)

	with open("weights.json") as f:
		weights = json.load(f)

	data = fill_nan_with_mean(dataset)
	values = get_values(data)
	predictions2 = predict(values, weights)
	expected2 = dataset["Hogwarts House"].tolist()

	accuracy = (1 / len(expected2)) * sum([1 if expected2[i] == predictions2[i] else 0 for i in range(len(expected2))])
	# accuracy = accuracy_score(expected, predictions)

	print(f"Accuracy: {100*accuracy:.2f}%")


def main():
	dataset = load('datasets/dataset_test.csv')
	if dataset is None:
		exit(1)

	with open("weights.json") as f:
		weights = json.load(f)

	data = fill_nan_with_mean(dataset)
	values = get_values(data)
	predictions = predict(values, weights)
	save_in_csv_file(predictions)

	compare_predictions(predictions)


if __name__ == "__main__":
	main()
