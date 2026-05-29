import pandas as pd
import numpy as np
import math
import json
import csv


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


if __name__ == "__main__":

	dataset = load('datasets/dataset_test.csv')
	if dataset is None:
		exit(1)

	with open("weights.json") as f:
		weights = json.load(f)


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

	# pour chaque étudiant
	# calculer la probabilité d'appartenir à chaque maison
	# choisir la maison avec la probabilité la plus élevée
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

	# écriture des résultats dans un fichier CSV
	with open("houses.csv", "w", newline="") as f:

		writer = csv.writer(f)

		writer.writerow(["Index", "Hogwarts House"])

		for i, pred in enumerate(predictions):
			writer.writerow([i, pred])







	# tests simplifiés avec seulement les features 2 et 3
	# pour voir si nos résultats sont cohérents
	# 2 -> Astronomy et 3 -> Herbology

	# si feature 2 >= 0 et feature 3 >= 0 => Hufflepuff
	# si feature 2 >= 0 et feature 3 < 0 => Gryffindor
	# si feature 2 < 0 et feature 3 >= 0 => Ravenclaw
	# si feature 2 < 0 et feature 3 < 0 => Slytherin
	
	simplified = []

	for i in range(len(values)):
		if values[i][1] >= 0 :
			if values[i][2] >= 0 :
				simplified.append("Hufflepuff")
			else:
				simplified.append("Gryffindor")
		else:
			if values[i][2] >= 0 :
				simplified.append("Ravenclaw")
			else:
				simplified.append("Slytherin")

	count = 0

	for i in range(len(predictions)):
		if predictions[i] != simplified[i]:
			count += 1
			print(f"Index {i}: predicted {predictions[i]}, simplified {simplified[i]}")

	print(f"Number of differences: {count}")

	# 19 différences sur 400 étudiants
	# soit environ ~5% de différences entre les prédictions du modèle et la classification simplifiée
	# les resultats sont cohérents
	# maintenant il faut analyser la précision et rendre le modèle plus précis si nécessaire

