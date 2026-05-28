import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


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


if __name__ == "__main__":

	features = [
		"Arithmancy",
		"Astronomy",
		"Herbology",
		"Defense Against the Dark Arts",
		"Divination",
		"Muggle Studies",
		"Ancient Runes",
		"History of Magic",
		"Transfiguration",
		"Potions",
		"Care of Magical Creatures",
		"Charms",
		"Flying"
	]

	dataset = load('datasets/dataset_train.csv')
	if dataset is None:
		exit(1)

	data = np.array(dataset)

	fig, axs = plt.subplots(4, 4, figsize=(10, 8))

	for n in range(len(features)):

		ligne = n // 4
		colonne = n % 4

		gryffindor = []
		ravenclaw = []
		hufflepuff = []
		slytherin = []

		for i in range(len(data)):
			tmp = None
			if data[i][1] == "Gryffindor":
				tmp = gryffindor
			elif data[i][1] == "Ravenclaw":
				tmp = ravenclaw
			elif data[i][1] == "Hufflepuff":
				tmp = hufflepuff
			elif data[i][1] == "Slytherin":
				tmp = slytherin

			if not math.isnan(float(data[i][n + 6])):
				tmp.append(float(data[i][n + 6]))

		axs[ligne, colonne].hist(gryffindor, bins=5, alpha=0.5, color='red', label="gryffindor")
		axs[ligne, colonne].hist(ravenclaw, bins=5, alpha=0.5, color='blue', label="ravenclaw")
		axs[ligne, colonne].hist(hufflepuff, bins=5, alpha=0.5, color='yellow', label="hufflepuff")
		axs[ligne, colonne].hist(slytherin, bins=5, alpha=0.5, color='green', label="slytherin")

		axs[ligne, colonne].set_title(features[n])

	fig.delaxes(axs[3, 1])
	fig.delaxes(axs[3, 2])
	fig.delaxes(axs[3, 3])

	plt.tight_layout()
	plt.show()
