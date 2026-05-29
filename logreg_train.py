import pandas as pd
import numpy as np
import math
import json


LEARNING_RATE = 0.01
EPOCHS = 1000


def load(path: str) -> pd.DataFrame:
	"""Load a CSV file into a map object."""

	try:

		dataset = pd.read_csv(path)

		return dataset

	except FileNotFoundError:
		print("Error: File not found.")
		return None

	except pd.errors.EmptyDataError:
		print("Error: File empty or invalid.")
		return None

	except pd.errors.ParserError:
		print("Error: File format invalid.")
		return None

	except Exception as e:
		print(f"Error: An unexpected error occurred: {e}")
		return None


def standardize(X):
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)

	X_scaled = (X - mean) / std

	return X_scaled


def sigmoid(z) -> float:
	return 1 / (1 + np.exp(-z))


def train_binary_classifier(X, y, lr=LEARNING_RATE, epochs=EPOCHS):

	theta = np.zeros(13)

	for _ in range(epochs):

		h = sigmoid(X @ theta)

		gradient = (1 / len(X)) * (np.transpose(X) @ (h - y))

		theta -= lr * gradient

	return theta


if __name__ == "__main__":

	dataset = load('datasets/dataset_train.csv')
	if dataset is None:
		exit(1)

	data = np.array(dataset)

	# remplacement des NaN par la moyenne de chaque colonne
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

	# récupération des valeurs numeriques (les notes)
	values = [[0.0 for _ in range(13)] for _ in range(len(data))]

	for i in range(len(data)):
		for col in range(13):
			values[i][col] = float(data[i][col + 6])

	values = standardize(values)

	houses = [
		"Gryffindor",
		"Hufflepuff",
		"Ravenclaw",
		"Slytherin"
	]

	weights = {}

	# entrainement d'un classifieur binaire pour chaque maison
	for house in houses:

		y = dataset["Hogwarts House"]
		y_binary = (y == house).astype(int)

		theta = train_binary_classifier(np.array(values), y_binary)

		weights[house] = theta.tolist()

	# écriture des poids dans un fichier JSON
	with open("weights.json", "w") as f:
		json.dump(weights, f)
