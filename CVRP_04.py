# Building CVRP solver using metaheuristics from scratch.

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import time

def read_txt_file(filename):
    """
    Read coordinates and demand values from a .txt file.
    Assumes the data is in the format: X Y Demand.
    """
    custnum     = []
    coordx      = []
    coordy      = []
    demand      = []
    readytime   = []
    duedate     = []
    servicetime = []

    count = 0

    # Our data file has a header, so we skip it; and also has titles for the number of vehicles and the capacity.
    with open(filename, 'r') as file:
        for line in file:
            elements = line.strip().split()
            if count == 4:
                veh_num = int(elements[0])
                capacity = int(elements[1])
                # Now, from line 9, we have the data we need for the model.
            if count >= 9:
                custnum.append(     int     (elements[0]))  # Customer number
                coordx.append(      float   (elements[1]))  # X coordinate of each customer
                coordy.append(      float   (elements[2]))  # Y coordinate of each customer
                demand.append(      int     (elements[3]))  # Demand of each customer
                readytime.append(   float   (elements[4]))  # Ready time of each customer (won't be using it for CVRP)
                duedate.append(     float   (elements[5]))  # Due date of each customer (won't be using it for CVRP)
                servicetime.append( float   (elements[6]))  # Service time of each customer

                """class Client[count-9]:
                    def __init__(self, custnum, coordx, coordy, demand, readytime, duedate, servicetime):
                        self.custnum     = custnum[count-9]
                        self.coordx      = coordx[count-9]
                        self.coordy      = coordy[count-9]
                        self.demand      = demand[count-9]
                        self.readytime   = readytime[count-9]
                        self.duedate     = duedate[count-9]
                        self.servicetime = servicetime[count-9]"""

            count += 1
    
    # extracted_data = [veh_num, capacity, custnum, coordx, coordy, demand, readytime, duedate, servicetime]

    return veh_num, capacity, custnum, coordx, coordy, demand, readytime, duedate, servicetime

# Vamos a usar la distancia Euclidea entre dos ciudades
def distance_euclidean_AB(A, B):
    return ((A[0] - B[0])**2 + (A[1] - B[1])**2) ** 0.5

# Funcion para calcular la distancia entre dos ciudades
def distance(city1, city2):
    # Aqui es donde cambio el tipo de distancia que quiero usar. En este fichero solo tenemos disponible la euclidea
    return distance_euclidean_AB(city1, city2)

# Funcion para calcular la distancia total de una ruta
def total_distance(route, diction):
    total = 0
    for i in range(len(route) - 1):
        total += distance(diction[route[i]], diction[route[i+1]])
    return total

# We are meant to change this objective function according to CVRP optimization objectives
def objective_function(solution, diction):
    return total_distance(solution, diction)

# Distance matrix makes it computationally faster to calculate the distance between two points.
def calculate_distance_matrix(coordinates):
    """
    Calculate the distance matrix between coordinates.
    """
    num_points = len(coordinates)
    dist_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i, j] = calculate_distance(coordinates, i, j)

    return dist_matrix

# Review if distance is correctly calculated.
# This function works pretty much the same as distance_euclidean_AB
def calculate_distance(coordinates, i, j):
    """
    Calculate the Euclidean distance between two points.
    """
    x1, y1 = coordinates[i]
    x2, y2 = coordinates[j]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Review if total distance is correctly calculated
def calculate_total_distance(route, dist_matrix):
    """
    Calculate the total distance of a given route using the distance matrix.
    """
    total_distance = 0
    num_points = len(route)

    for i in range(num_points - 1):
        current_node = route[i]
        next_node = route[i + 1]
        total_distance += dist_matrix[current_node, next_node]

    return total_distance

def plot_problem(dic_customers: dict):
    customers = list(dic_customers.keys()) # List of customers to visit
    X = [dic_customers[customer][0] for customer in customers] # List of X coordenates
    Y = [dic_customers[customer][1] for customer in customers] # List of Y coordenates
    X.append(X[0]) # Initial position X 
    Y.append(Y[0]) # Initial position Y
    plt.plot(X,Y,'o')
    for customer in customers:  # Show customer number on each customer
        plt.annotate(customer, (dic_customers[customer][0]+0.01, dic_customers[customer][1]))

# Have in mind: auxiliary functions shall be put in a separate file for cleanness.

# Main function
def main():

    veh_num, capacity, custnum, coordx, coordy, demand, readytime, duedate, servicetime = read_txt_file("C:\\Users\\Pablo\\OneDrive - UNIVERSIDAD DE SEVILLA\\MOIGE\\CUATRI2\\MOPG\\Python\\07 CVRP\\CVRP\\Datos.txt")

    dict_xy = {custnum[i]: (coordx[i], coordy[i]) for i in range(len(custnum))}

    # Plotting the problem
    plot_problem(dict_xy)
    plt.show()
    

# Run the main function
if __name__ == '__main__':
    main()
