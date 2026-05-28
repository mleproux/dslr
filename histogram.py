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

	print("Features:")
	for i in range(len(features)):
		print(f"{i} - {features[i]}")

	col = input("Enter the feature number to plot (0-12): ")
	try:
		col = int(col)
	except ValueError:
		print("Error: Invalid feature number.")
		exit(1)
	if col < 0 or col >= len(features):
		print("Error: Invalid feature number.")
		exit(1)

	print(f"Plotting histogram for feature: {features[col]}")

	data = np.array(dataset)

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

		if not math.isnan(float(data[i][col + 6])):
			tmp.append(float(data[i][col + 6]))

	plt.hist(gryffindor, bins=5, alpha=0.5, label="gryffindor")
	plt.hist(ravenclaw, bins=5, alpha=0.5, label="ravenclaw")
	plt.hist(hufflepuff, bins=5, alpha=0.5, label="hufflepuff")
	plt.hist(slytherin, bins=5, alpha=0.5, label="slytherin")

	plt.xlabel("Notes")
	plt.ylabel("Fréquence")
	plt.title(features[col])

	plt.legend()
	plt.show()

# 10 - Care of Magical Creatures
# 0 - Arithmancy ~
