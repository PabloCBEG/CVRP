# Building CVRP solver using metaheuristics from scratch.

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

def objective_function(solution, diction):
    return total_distance(solution, diction)
    



# Main function
def main():

    veh_num, capacity, custnum, coordx, coordy, demand, readytime, duedate, servicetime = read_txt_file("C:\\Users\\Pablo\\OneDrive - UNIVERSIDAD DE SEVILLA\\MOIGE\\CUATRI2\\MOPG\\Python\\07 CVRP\\CVRP\\Datos.txt")

    dict_xy = {custnum[i]: (coordx[i], coordy[i]) for i in range(len(custnum))}
    

# Run the main function
if __name__ == '__main__':
    main()
