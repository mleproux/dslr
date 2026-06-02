import pandas as pd
import matplotlib.pyplot as plt
import sys


features = (
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
)


house_colors = {
    "Gryffindor": "red",
    "Hufflepuff": "yellow",
    "Ravenclaw": "blue",
    "Slytherin": "green"
}


def get_data(file_name: str):
    """Load dataset from a CSV file.
    """
    data = pd.read_csv(file_name)
    return data
            

def main():
    """Main program :
    - Load dataset from CSV file
    - Export a pair plot of every feature as a PNG file
    """
    argc: int = len(sys.argv)
    if argc != 2:
        print("Error: Invalid arguments")
        return 0
    
    file_name: str = sys.argv[1]
    
    try:
        data = get_data(file_name)
        courses = data[list(features)]
        num_features = len(features)
        colors = data["Hogwarts House"].map(house_colors)
        
        fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))
        
        for i in range(num_features):
            for j in range(num_features):
                ax = axes[i, j]
                if i == j:
                    for house, color in house_colors.items():
                        mask = data["Hogwarts House"] == house
                        ax.hist(courses.iloc[:, i][mask], bins=15, alpha=0.4, color=color)
                else:
                    ax.scatter(courses.iloc[:, j], courses.iloc[:, i], s=1, c=colors)
                if j == 0:
                    ax.set_ylabel(courses.columns[i], fontsize=8)
                if i == num_features - 1:
                    ax.set_xlabel(courses.columns[j], fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig("pair_plot.png", dpi=300)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as error:
        print(Exception.__name__ + ":", error)


if __name__ == '__main__' :
	main()