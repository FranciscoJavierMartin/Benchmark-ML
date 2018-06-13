import random

if __name__ == "__main__":
    number_of_points = 10000
    number_of_features = 40

    probability_of_empty_values = 0.1

    output_file_name ="data/clustering.gauss." + str(number_of_points)+"."+ str(number_of_features)+".txt"

    random.seed(1)

    with open(output_file_name, "w") as file:
        for point in range(number_of_points):
            file.write(str(point) + " ")
            for feature in range(number_of_features):
                if (random.random() > probability_of_empty_values):
                    value = random.gauss(0, 1) #random
                    file.write(str(feature+1)+":"+str(value)+ " ")
            file.write("\n")

    file.close()
