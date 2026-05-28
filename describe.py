import pandas as pd
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

	dataset = load('datasets/dataset_train.csv')
	if dataset is None:
		exit(1)

	data = np.array(dataset)
	tab = []

	for i in range(13):
		col = []
		for j in range(len(data)):
			if not math.isnan(data[j][i + 6]):
				col.append(data[j][i + 6])
		col = np.array(col)

		total = float(0.0)
		min_val = float(0.0)
		max_val = float(0.0)

		for elem in col:
			total += elem
			if elem < min_val:
				min_val = elem
			if elem > max_val:
				max_val = elem

		count = len(col)
		mean = total / count
		std = math.floor(math.sqrt(sum((elem - mean) ** 2 for elem in col) / count) * 100) / 100

		sorted_col = sorted(col)
		q1 = math.floor(sorted_col[int((count + 1) / 4)] * 100) / 100
		q2 = math.floor(sorted_col[int((count + 1) / 2)] * 100) / 100
		q3 = math.floor(sorted_col[int(3 * (count + 1) / 4)] * 100) / 100

		mean = math.floor(mean * 100) / 100
		min_val = math.floor(min_val * 100) / 100
		max_val = math.floor(max_val * 100) / 100

		feature = np.array([count, mean, std, min_val, q1, q2, q3, max_val])
		tab.append(feature)

	print("\tFeature 1\tFeature 2\tFeature 3\tFeature 4\tFeature 5\tFeature 6\tFeature 7\tFeature 8\tFeature 9\tFeature 10\tFeature 11\tFeature 12\tFeature 13")
	print(f"Count\t{tab[0][0]}\t\t{tab[1][0]}\t\t{tab[2][0]}\t\t{tab[3][0]}\t\t{tab[4][0]}\t\t{tab[5][0]}\t\t{tab[6][0]}\t\t{tab[7][0]}\t\t{tab[8][0]}\t\t{tab[9][0]}\t\t{tab[10][0]}\t\t{tab[11][0]}\t\t{tab[12][0]}")
	print(f"Mean\t{tab[0][1]}\t{tab[1][1]}\t\t{tab[2][1]}\t\t{tab[3][1]}\t\t{tab[4][1]}\t\t{tab[5][1]}\t\t{tab[6][1]}\t\t{tab[7][1]}\t\t{tab[8][1]}\t\t{tab[9][1]}\t\t{tab[10][1]}\t\t{tab[11][1]}\t\t{tab[12][1]}")
	print(f"Std\t{tab[0][2]}\t{tab[1][2]}\t\t{tab[2][2]}\t\t{tab[3][2]}\t\t{tab[4][2]}\t\t{tab[5][2]}\t\t{tab[6][2]}\t\t{tab[7][2]}\t\t{tab[8][2]}\t\t{tab[9][2]}\t\t{tab[10][2]}\t\t{tab[11][2]}\t\t{tab[12][2]}")
	print(f"Min\t{tab[0][3]}\t{tab[1][3]}\t\t{tab[2][3]}\t\t{tab[3][3]}\t\t{tab[4][3]}\t\t{tab[5][3]}\t\t{tab[6][3]}\t\t{tab[7][3]}\t\t{tab[8][3]}\t\t{tab[9][3]}\t\t{tab[10][3]}\t\t{tab[11][3]}\t\t{tab[12][3]}")
	print(f"25%\t{tab[0][4]}\t\t{tab[1][4]}\t\t{tab[2][4]}\t\t{tab[3][4]}\t\t{tab[4][4]}\t\t{tab[5][4]}\t\t{tab[6][4]}\t\t{tab[7][4]}\t\t{tab[8][4]}\t\t{tab[9][4]}\t\t{tab[10][4]}\t\t{tab[11][4]}\t\t{tab[12][4]}")
	print(f"50%\t{tab[0][5]}\t\t{tab[1][5]}\t\t{tab[2][5]}\t\t{tab[3][5]}\t\t{tab[4][5]}\t\t{tab[5][5]}\t\t{tab[6][5]}\t\t{tab[7][5]}\t\t{tab[8][5]}\t\t{tab[9][5]}\t\t{tab[10][5]}\t\t{tab[11][5]}\t\t{tab[12][5]}")
	print(f"75%\t{tab[0][6]}\t\t{tab[1][6]}\t\t{tab[2][6]}\t\t{tab[3][6]}\t\t{tab[4][6]}\t\t{tab[5][6]}\t\t{tab[6][6]}\t\t{tab[7][6]}\t\t{tab[8][6]}\t\t{tab[9][6]}\t\t{tab[10][6]}\t\t{tab[11][6]}\t\t{tab[12][6]}")
	print(f"Max\t{tab[0][7]}\t{tab[1][7]}\t\t{tab[2][7]}\t\t{tab[3][7]}\t\t{tab[4][7]}\t\t{tab[5][7]}\t\t{tab[6][7]}\t\t{tab[7][7]}\t\t{tab[8][7]}\t\t{tab[9][7]}\t\t{tab[10][7]}\t\t{tab[11][7]}\t\t{tab[12][7]}")


# idee bonus : mettre resultats dans .csv ...
