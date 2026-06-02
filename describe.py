import pandas as pd
import numpy as np
import math


houses = [
	"Gryffindor",
	"Ravenclaw",
	"Hufflepuff",
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


def bonus_field(data, i, house):
	"""Calculate the bonus field based on the given column.
	"""
	count = 0
	total = 0.0
	for k in range(len(data)):
		if data[k][1] == house and not math.isnan(data[k][i + 6]):
			min = data[k][i + 6]
			max = data[k][i + 6]
			break
	for k in range(len(data)):
		if data[k][1] == house and not math.isnan(data[k][i + 6]):
			count += 1
			total += data[k][i + 6]
			if data[k][i + 6] < min:
				min = data[k][i + 6]
			if data[k][i + 6] > max:
				max = data[k][i + 6]
	mean = math.floor(total / count * 100) / 100
	min = math.floor(min * 100) / 100
	max = math.floor(max * 100) / 100
	if math.fabs(mean) > 1000:
		mean = math.floor(mean * 10) / 10
	if math.fabs(min) > 1000:
		min = math.floor(min * 10) / 10
	if math.fabs(max) > 1000:
		max = math.floor(max * 10) / 10
	return [count, mean, min, max]


def print_house(tab, n):
	"""Print the bonus fields for a specific house.
	"""
	str = "\t\t"
	if n == 3:
		str = "\t"
	bonus_fields = ["Count", "Mean", "Min", "Max"]
	for i in range(4):
		str = "\t\t"
		if n == 3 and i >= 2:
			str = "\t"
		print(f"{bonus_fields[i]}\t{tab[0][4 * n + i]}{str}{tab[1][4 * n + i]}\t\t{tab[2][4 * n + i]}\t\t{tab[3][4 * n + i]}\t\t{tab[4][4 * n + i]}\t\t{tab[5][4 * n + i]}\t\t{tab[6][4 * n + i]}\t\t{tab[7][4 * n + i]}\t\t{tab[8][4 * n + i]}\t\t{tab[9][4 * n + i]}\t\t{tab[10][4 * n + i]}\t\t{tab[11][4 * n + i]}\t\t{tab[12][4 * n + i]}")


def print_bonus(tab):
	"""Print the bonus fields for each house.
	"""
	for i in range(4):
		print(houses[i])
		print_house(tab, i + 2)


def print_results(tab):
	"""Print the results in a formatted way.
	"""
	print("\tFeature 1\tFeature 2\tFeature 3\tFeature 4\tFeature 5\tFeature 6\tFeature 7\tFeature 8\tFeature 9\tFeature 10\tFeature 11\tFeature 12\tFeature 13")
	fields = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
	for i in range(8):
		if i in [1, 2, 3, 7]:
			str = "\t"
		else:
			str = "\t\t"
		print(f"{fields[i]}\t{tab[0][i]}{str}{tab[1][i]}\t\t{tab[2][i]}\t\t{tab[3][i]}\t\t{tab[4][i]}\t\t{tab[5][i]}\t\t{tab[6][i]}\t\t{tab[7][i]}\t\t{tab[8][i]}\t\t{tab[9][i]}\t\t{tab[10][i]}\t\t{tab[11][i]}\t\t{tab[12][i]}")
	print_bonus(tab)


def main():
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
		min_val = float(col[0])
		max_val = float(col[0])

		for elem in col:
			total += elem
			if elem < min_val:
				min_val = elem
			if elem > max_val:
				max_val = elem

		count = len(col)
		mean = total / count
		std = math.floor(math.sqrt(sum((elem - mean) ** 2 for elem in col) / count) * 100) / 100
		mean = math.floor(mean * 100) / 100
		min_val = math.floor(min_val * 100) / 100
		max_val = math.floor(max_val * 100) / 100
		sorted_col = sorted(col)
		q1 = math.floor(sorted_col[int((count + 1) / 4)] * 100) / 100
		q2 = math.floor(sorted_col[int((count + 1) / 2)] * 100) / 100
		q3 = math.floor(sorted_col[int(3 * (count + 1) / 4)] * 100) / 100

		feature = np.array([count, mean, std, min_val, q1, q2, q3, max_val])
		for house in houses:
			bonus_fields = bonus_field(data, i, house)
			for field in bonus_fields:
				feature = np.append(feature, field)
		tab.append(feature)
	print_results(tab)


if __name__ == "__main__":
	main()
 