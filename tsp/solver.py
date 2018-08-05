#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from queue import PriorityQueue
from sklearn.neighbors import NearestNeighbors
from numba import jit
from io import StringIO

class Vertex:
    
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y

class Distance(object):

    def __init__(self, solution, distance):
        self.solution = solution
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance
    
class VertexDistance(object):

    def __init__(self, index, distance):
        self.index = index
        self.distance = distance
        
    def __lt__(self, other):
        return self.distance < other.distance

def euclidean_distance(vertex_1, vertex_2):
    return np.sqrt( np.square(vertex_1.x - vertex_2.x)  + np.square(vertex_1.y - vertex_2.y) )

def objective_function(route):
    """
    Assume vertices is route starting and ending at same vertex e.g.
    [1, 3, 2, 0, 1]
    """
    head_route = route[0:(len(route)-1)]
    tail_route = route[1:]
    combined_route = zip(head_route, tail_route)
    dist_sum = 0.0
    for vertex_pair in combined_route:
        dist_sum += euclidean_distance(vertex_pair[0], vertex_pair[1])
    return dist_sum
    
def is_valid_hamiltonian_cycle(route, n):
    # Ensure it starts and ends at same node
    start_and_end = route[0] == route[-1]
    
    # Need to ensure all vertices 0,..,n-1 are present
    all_index = list(range(0, n))
    indexes = sorted(list(set([vertex.index for vertex in route])))
    index_present = all_index == indexes
    
    return start_and_end and index_present

def tour_string(vertices):
    return " ".join([str(i.index) for i in vertices])

def tour_list(vertices):
    return [i.index for i in vertices]

# Improved NN approach
@jit
def nearest_neighbour_improved(vertices, n, points, max_search=0):
    nn_max = min(n, 500)
    remaining_indexes = set(range(0, n))
    starting_vertex = np.random.randint(0, n)
    remaining_indexes.remove(starting_vertex)
    tour = [vertices[starting_vertex]]
    used_index = set([starting_vertex])
    
    nbrs = NearestNeighbors(n_neighbors=nn_max, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    while len(remaining_indexes) > 0:
        cur_vertex = tour[-1]
        # Search nearest neighbours first
        nn_index = indices[cur_vertex.index]
        found_in_nearest = False
        for i in nn_index[1:]:
            if i not in used_index:
                # Found nearest, update the tour
                tour.append(vertices[i])
                remaining_indexes.remove(i)
                    
                used_index.add(i)
                found_in_nearest = True
                break
                
        if not found_in_nearest:
            solution_queue = PriorityQueue()

            num_checked = 0
            for i in remaining_indexes:
                dist = euclidean_distance(cur_vertex, vertices[i])
                solution_queue.put(VertexDistance(i, dist))
                num_checked += 1

                if (max_search != 0 and num_checked > max_search):
                    break

            # Select the shortest distance
            nearest = solution_queue.get()

            # Update the tour
            tour.append(vertices[nearest.index])
            remaining_indexes.remove(nearest.index)
            used_index.add(nearest.index)
            
    # Add final node to tour
    tour.append(tour[0])
        
    return tour
            
@jit
def nearest_neighbour(vertices, n, max_search=0):
    remaining_indexes = set(range(0, n))
    starting_vertex = np.random.randint(0, n)
    remaining_indexes.remove(starting_vertex)
    tour = [vertices[starting_vertex]]
    
    while len(remaining_indexes) > 0:
        cur_vertex = tour[-1]
        solution_queue = PriorityQueue()
        
        num_checked = 0
        for i in remaining_indexes:
            dist = euclidean_distance(cur_vertex, vertices[i])
            solution_queue.put(VertexDistance(i, dist))
            num_checked += 1
            
            if (max_search != 0 and num_checked > max_search):
                break
            
        # Select the shortest distance
        nearest = solution_queue.get()
        
        # Update the tour
        tour.append(vertices[nearest.index])
        remaining_indexes.remove(nearest.index)
        
    # Add final node to tour
    tour.append(tour[0])
        
    return tour

@jit
def two_opt_rand(vertices, n, points):
    
    route = nearest_neighbour_improved(vertices, n, points)
    cur_distance = objective_function(route)
    while True:
        print(cur_distance)

        # Randomely pick an edge
        i = np.random.randint(1, n)
       
        solution_queue = PriorityQueue()
        for j in range(i+1, len(route)):
            new_route = route[:]
            new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
            
            assert(is_valid_hamiltonian_cycle(new_route, n))
            value = objective_function(new_route)
            solution_queue.put(Distance(new_route, value))
            
        best_move = solution_queue.get()
        best_distance = best_move.distance
        
        if (best_distance < cur_distance):
            cur_distance = best_distance
            route = best_move.solution
        else:
            break
            
    return route

@jit
def two_opt_full(vertices, n, points, seconds_timeout = 60):
        
    route = nearest_neighbour_improved(vertices, n, points)
    cur_distance = objective_function(route)
    start_time = round(time.time())
    
    while True:
        print(cur_distance)
        
        # Check timeout
        time_now = round(time.time())
        time_diff = time_now - start_time
        if (seconds_timeout != -1 and time_diff > seconds_timeout):
            print("TIMEOUT")
            break

        # Randomely pick an edge
        solution_queue = PriorityQueue()
        for i in range(1, n):
            # Check timeout
            time_now = round(time.time())
            time_diff = time_now - start_time
            if (seconds_timeout != -1 and time_diff > seconds_timeout):
                print("TIMEOUT")
                break
            
            for j in range(i+1, len(route)):
                new_route = route[:]
                new_route[i:j] = route[j-1:i-1:-1] # this is the 2-opt Swap
    
                assert(is_valid_hamiltonian_cycle(new_route, n))
                value = objective_function(new_route)
                solution_queue.put(Distance(new_route, value))

        best_move = solution_queue.get()
        best_distance = best_move.distance

        if (best_distance < cur_distance):
            cur_distance = best_distance
            route = best_move.solution
        else:
            break
            
    return route

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    data_file = StringIO(input_data)
    data = pd.read_csv(data_file, sep=" ", names=["x", "y"], dtype={"x":float, "y":float})

    n = int(data["x"][0])
    print(n)
    vertices = []
    points = []
    for i in range(1, n+1):
        vertices.append(Vertex(i-1, data["x"][i], data["y"][i]))
        points.append((data["x"][i], data["y"][i]))

    solution = two_opt_full(vertices, n, points, 180)
    distance = objective_function(solution)

    # prepare the solution in the specified output format
    output_data = str(distance) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, tour_list(solution)[:-1]))
    return output_data

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

