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


def print_features():
    """Display the list of available features.
    """
    
    print("Features:")
    for i in range(len(features)):
        print(f"({i}) {features[i]}")


def get_feature():
    """Prompt the user to select a feature for plotting.
    """
    while(True):  
        try:
            user_input = input("Which feature do you want to scatter :\n")
            choice = int(user_input)
            if choice < 0 or choice > len(features) - 1:
                    raise ValueError
            break
        except ValueError:
            print(f"Incorrect input. Please provide a valid input (1-{len(features) - 1}).")
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
            sys.exit(0)
    return features[choice]
            

def plot_points(data, x_label, y_label, colors):
    """Create a scatter plot for two selected features.
    """
    x = data[x_label]
    y = data[y_label]
    
    plt.scatter(x, y, c=colors)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    

def main():
    """Main program :
    - Load dataset from CSV file
    - Ask the user to select two features
    - Display the scatter plot
    """
    argc: int = len(sys.argv)
    if argc != 2:
        print("Error: Invalid arguments")
        return 0
    
    file_name: str = sys.argv[1]
    print_features()
    x_label: str = get_feature()
    y_label: str = get_feature()
    
    try:
        data = get_data(file_name)
        data.dropna()
        colors = data["Hogwarts House"].map(house_colors)
        plot_points(data, x_label, y_label, colors)
        plt.show()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as error:
        print(Exception.__name__ + ":", error)


if __name__ == '__main__' :
	main()