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

def main():

    veh_num, capacity, custnum, coordx, coordy, demand, readytime, duedate, servicetime = read_txt_file("C:\\Users\\Pablo\\OneDrive - UNIVERSIDAD DE SEVILLA\\MOIGE\\CUATRI2\\MOPG\\Python\\07 CVRP\\CVRP\\Datos.txt")

    for i in range(len(demand)):
        print("demand:\n", demand[i])
    
    print("Len:\n", len(demand))

# Run the main function
if __name__ == '__main__':
    main()
