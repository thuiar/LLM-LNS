import numpy as np
import networkx as nx
# Read coordinates from input file
from numba import jit
import os

#@jit(nopython=True) 
def read_coordinates(instance_path, file_name):
    # print("file_name = ", file_name)
    coordinates = []
    optimal_distance = 1E10

    file = open(instance_path + "/" + file_name, 'r')
    lines = file.readlines()
    for line in lines:
        if line.startswith('NODE_COORD_SECTION'):
            index = lines.index(line) + 1
            break
    for i in range(index, len(lines)-1):
        parts = lines[i].split()
        # print("parts:", parts)
        if (parts[0]=='EOF'): break

        coordinates.append([float(parts[1]), float(parts[2])])
    sol = open(instance_path + "/solutions", 'r')
    lines = sol.readlines()
    for line in lines:
        if line.startswith(file_name.removesuffix(".tsp")):
            optimal_distance = float(line.split()[2])
            break

    return np.array(coordinates), optimal_distance

#@jit(nopython=True) 
def create_distance_matrix(coordinates):

    x = np.array([coord[0] for coord in coordinates])
    y = np.array([coord[1] for coord in coordinates])
    distance_matrix = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
    return distance_matrix

#@jit(nopython=True) 
def read_instance(instance_path, filename):

    # Test the code
    coord, opt_cost = read_coordinates(instance_path, filename)
    # print("coord:", coord)
    instance = create_distance_matrix(coord)

    return coord, instance, opt_cost

def read_instance_all(instances_path):
    # print("!!!")
    file_names = os.listdir(instances_path)
    coords = []
    instances = []
    opt_costs = []
    names = []
    for filename in file_names:
        if filename.endswith('.tsp'):
            coord, instance, opt_cost = read_instance(instances_path, filename)
            coords.append(coord)
            instances.append(instance)
            opt_costs.append(opt_cost)
            names.append(filename)
    return coords, instances, opt_costs, names



if __name__ == '__main__':
    G,scale= read_instance_all('../TSPLib200/eil51.tsp')
    print(G.edges[0,0]['weight'])
    print(scale)