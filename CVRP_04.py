# Building CVRP solver using metaheuristics from scratch.

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

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
    for customer in customers:
        plt.annotate(customer, (dic_customers[customer][0]+0.01, dic_customers[customer][1]))

def plot_solution(solution: list, dic_customers: dict):
    for route in solution:
        X = [dic_customers[customer][0] for customer in route]
        Y = [dic_customers[customer][1] for customer in route]
        X.append(X[0])
        Y.append(Y[0])
        plt.plot(X,Y)

# Funcion para intercambiar dos ciudades en una ruta
def swap(route, i, j):
    route[i], route[j] = route[j], route[i]

# Funcion para generar una vecindad SWAP a partir de una ruta
def neighbourhood_swap(current_solution):
    neighbourhood = []
    for i in range(len(current_solution)):
        for j in range(i+1, len(current_solution)):
            new_route = current_solution.copy()
            swap(new_route, i, j)
            neighbourhood.append(new_route)
    return neighbourhood

# Funcion para generar una vecindad INSERTION a partir de una ruta
def neighbourhood_insertion(current_solution):
    neighbourhood = []
    for i in range(len(current_solution)):
        for j in range(len(current_solution)):
            if i != j:
                neighbour = current_solution[:]
                neighbour.remove(current_solution[i])
                neighbour.insert(j, current_solution[i])
                neighbourhood.append(neighbour)
    return neighbourhood

vswap = []
vinsertion = []
neighbourhood_list = [vswap, vinsertion]
umax = len(neighbourhood_list)

current_distance_array = []
color_plot_array = ['r', 'g', 'b', 'k', 'y', 'p', 'm']
u_array_for_plotting = []
label_list_for_plotting = ['Swap neighbourhood', 'Insertion neighbourhood','Last found solution']

def vnd_for_cvrp(route, dict_xy):

    current_route = route.copy()
    best_route = route.copy()
    best_distance = objective_function(current_route, dict_xy)
    available_neigbourhoods = [neighbourhood_swap, neighbourhood_insertion]

    u = 1                                                                           # u es el numero del vecindario en que estoy buscando
    while u <= umax:                                                                # en este caso tenemos solo 2 vecindarios, swap (1) e insertion (2)
                                                                                    # de manera que umax = 2
        neighbourhood_list[u-1] = available_neigbourhoods[u-1](current_route)       # genero el vecindario correspondiente

        current_route = min(neighbourhood_list[u-1], key=(lambda x: objective_function(x, dict_xy)))
        current_distance = objective_function(current_route, dict_xy)

        current_distance_array.append(round(current_distance,2))

        if current_distance < best_distance:
            best_route = current_route.copy()
            best_distance = current_distance
        
        else:
            u += 1

        u_array_for_plotting.append(u)

        # SI se quiere visualizar la evolucion de la solucion,
        # "descomentar" las 4 lineas posteriores. Habra que ir cerrando la ventana de plot para que se siga ejecutando
        # (esto permite apreciar como va cambiando)
        # plt.figure(3)
        # plot_problem(dict_xy)
        # plot_solution(current_route, dict_xy)
        # plt.show()

    print("01 Ha ejecutado correctamente VND\n")
    print("Ruta obtenida a partir de los clientes restantes: ", best_route, "\n")

    return best_route, best_distance

def cvrp_solver(route, dict_xy, dist_matrix, veh_num, capacity, demand):
    num_points = dist_matrix.shape[0]
    visited = np.zeros(num_points, dtype=bool)
    routes = []
    # route_aux = []
    current_node = 0
    carga = np.zeros(veh_num, dtype=int)
    # route = [current_node]
    visited[current_node] = True
    load_index = 0

    print("00 Ha entrado en cvrp_solver\n")

    while np.sum(visited) < num_points and len(routes) < veh_num:
        new_route, new_distance = vnd_for_cvrp(route, dict_xy)
        route_aux = [0]
        route_aux2 = []

        print("Tamaño de la nueva ruta: ", len(new_route), "\n")

        print("02 Ha ejecutado ", load_index, " veces el VND\n")

        for i in range(len(new_route)):
            if new_route[i] != 0:
                if carga[load_index] + demand[new_route[i]] <= capacity:
                    route_aux.append(new_route[i])
                    carga[load_index] += demand[new_route[i]]
                    visited[new_route[i]] = True
                else:
                    route_aux2 = route_aux.copy()
                    route_aux2.remove(0)
                    route_aux.append(0)
                    break
            else: i += 1
        
        load_index += 1
        routes.append(route_aux)

        # len_aux = len(route)
        # Remove visited clients from pending clients neighbourhood
        for i in range(len(route_aux2)):
            print("Elemento a eliminar: ", route_aux2[i], "\n")
            route.remove(route_aux2[i])

    return routes, carga

# Have in mind: auxiliary functions shall be put in a separate file for cleanness.

# Main function
def main():

    veh_num, capacity, custnum, coordx, coordy, demand, readytime, duedate, servicetime = read_txt_file("C:\\Users\\Pablo\\OneDrive - UNIVERSIDAD DE SEVILLA\\MOIGE\\CUATRI2\\MOPG\\Python\\07 CVRP\\CVRP\\Datos.txt")

    dict_xy = {custnum[i]: (coordx[i], coordy[i]) for i in range(len(custnum))}

    distanceMatrix = calculate_distance_matrix(list(zip(coordx, coordy)))

    initial_route = [i for i in range(len(custnum))]

    # print("Initial route: ", initial_route)

    # best_route, best_distance = vnd_for_cvrp(initial_route, dict_xy)

    routes, cargas = cvrp_solver(initial_route, dict_xy, distanceMatrix, veh_num, capacity, demand)

    # Plotting the problem
    # plot_problem(dict_xy)

    print("Vector de cargas: ", cargas)

    plt.figure(1)
    # plt.plot(range(0,len(current_distance_array)), current_distance_array, marker='o')
    for indice in range(0,len(current_distance_array)):
        plt.plot(indice, current_distance_array[indice], marker='o', color=color_plot_array[u_array_for_plotting[indice]-1])
    # label=label_list_for_plotting[u_array_for_plotting[indice]-1]
    plt.title('Convergencia a la mejor solucion obtenida')
    plt.figure(2)
    plot_problem(dict_xy)

    print("Routes:", routes)
    plot_solution(routes, dict_xy)

    plt.show()
    

# Run the main function
if __name__ == '__main__':
    main()
