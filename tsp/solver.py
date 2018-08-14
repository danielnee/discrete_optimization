#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import timeit

from queue import PriorityQueue
from sklearn.neighbors import NearestNeighbors
from numba import jit
from io import StringIO
from itertools import chain

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

def path_distance(r, c):
    r = [x.index for x in r]
    return np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])

def compute_distance_matrix(vertices, n):
    all_pairs = [(i, j) for i in range(0, n) for j in range(0, n) if i != j]
    dist_matrix = np.zeros((n, n))
    for k in all_pairs:
        x = k[0]; y = k[1]
        dist_matrix[x, y] = euclidean_distance(vertices[x], vertices[y])
    return dist_matrix

def compute_tour_distance(route, dist_matrix):
    head_route = route[:-1]
    tail_route = route[1:]
    pair_route = zip(head_route, tail_route)    
    return sum(map(lambda pair: dist_matrix[pair[0].index, pair[1].index], pair_route))

def test(vertex1, vertex2, dist_matrix):
    if (dist_matrix[vertex1.index, vertex2.index] == 0):
        dist_matrix[vertex1.index, vertex2.index] = euclidean_distance(vertex1, vertex2)
    return dist_matrix[vertex1.index, vertex2.index]

def compute_tour_distance_2(route, dist_matrix):
    head_route = route[:-1]
    tail_route = route[1:]
    pair_route = zip(head_route, tail_route)    
    #tour_sum = 0.0
    return sum(map(lambda pair: test(pair[0], pair[1], dist_matrix), pair_route))
    #for pairs in pair_route:
    #    vertex1 = pairs[0].index
    #    vertex2 = pairs[1].index
    #    if (dist_matrix[vertex1, vertex2] == 0):
    #        dist_matrix[vertex1, vertex2] = euclidean_distance(pairs[0], pairs[1])
    #    tour_sum += dist_matrix[vertex1, vertex2]
    #return tour_sum

def objective_function(route):
    """
    Assume vertices is route starting and ending at same vertex e.g.
    [1, 3, 2, 0, 1]
    """
    head_route = route[0:(len(route)-1)]
    tail_route = route[1:]
    combined_route = zip(head_route, tail_route)
    dist_sum = sum(list(map(lambda x: euclidean_distance(x[0], x[1]), combined_route)))
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
    
def two_opt_swap(i, j, route, n):
    new_route = route[:]
    new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
    return new_route

def two_opt_full(vertices, n, points, dist_matrix, seconds_timeout = 180):
        
    route = nearest_neighbour_improved(vertices, n, points)
    cur_distance = compute_tour_distance(route, dist_matrix)
    start_time = round(time.time())
    max_neighbour_size = 90000
    
    while True:
        print(cur_distance)
        
        # Check timeout
        time_now = round(time.time())
        time_diff = time_now - start_time
        if (seconds_timeout != -1 and time_diff > seconds_timeout):
            print("TIMEOUT")
            break

        neighbours = [(i, j) for i in range(1, n) for j in range(i+1, len(route)) if j - i > 1]
        routes = list(map(lambda x : two_opt_swap(x[0], x[1], route, n), neighbours))
        if len(routes) > max_neighbour_size:
            routes = random.sample(routes, max_neighbour_size)
        
        distances = list(map(lambda x : compute_tour_distance(x, dist_matrix), routes))

        y = np.argmin(distances)
        best_move = routes[y]
        best_distance = distances[y]

        if (best_distance < cur_distance):
            cur_distance = best_distance
            route = best_move
        else:
            break
            
    return route

def iterated_local_search(vertices, n, points, dist_matrix, max_searches = 3, seconds_timeout = 180):
    
    cur_route = nearest_neighbour_improved(vertices, n, points)
    cur_distance = compute_tour_distance(cur_route, dist_matrix)

    best_route = cur_route
    best_distance = cur_distance
    
    for i in range(0, max_searches):
    
        # Start at new random location each time 2opt is run
        cur_route =  tabu_search(vertices, n, points, dist_matrix, seconds_timeout)
        cur_distance = compute_tour_distance(cur_route, dist_matrix)

        if (cur_distance < best_distance):
            best_distance = cur_distance
            best_route = cur_route        
    
    return best_route

def tabu_search(vertices, n, points, dist_matrix, seconds_timeout = 180):
        
    route = nearest_neighbour_improved(vertices, n, points)
    cur_distance = compute_tour_distance(route, dist_matrix)
    start_time = round(time.time())
    max_neighbour_size = 20000
    tabu_list = np.zeros((n+1, n+1))
    L = 10
    it = 0
    
    while True:
        print(cur_distance)
        it += 1
        
        # Check timeout
        time_now = round(time.time())
        time_diff = time_now - start_time
        if (seconds_timeout != -1 and time_diff > seconds_timeout):
            print("TIMEOUT")
            break

        neighbours = [(i, j) for i in range(1, n) for j in range(i+1, len(route)) if j - i > 1 and 
                      tabu_list[i, j] <= it]
        print(len(neighbours))
        if len(neighbours) == 0:
            # reset
            tabu_list = np.zeros((n+1, n+1))
            it = 1
            neighbours = [(i, j) for i in range(1, n) for j in range(i+1, len(route)) if j - i > 1 and 
                      tabu_list[i, j] <= it]
        
        if len(neighbours) > max_neighbour_size:
            neighbours = random.sample(neighbours, max_neighbour_size)
        routes = list(map(lambda x : two_opt_swap(x[0], x[1], route, n), neighbours))    
        distances = list(map(lambda x : compute_tour_distance(x, dist_matrix), routes))
        # Update tabu_list
        for x in neighbours:
            tabu_list[x[0], x[1]] = tabu_list[x[0], x[1]] + L
        
        y = np.argmin(distances)
        best_move = routes[y]
        best_distance = distances[y]

        if (best_distance < cur_distance):
            cur_distance = best_distance
            route = best_move
        else:
            break
            
    return route

def tabu_search2(vertices, n, points, dist_matrix, seconds_timeout = 180):
        
    route = nearest_neighbour_improved(vertices, n, points)
    cur_distance = compute_tour_distance_2(route, dist_matrix)
    start_time = round(time.time())
    max_neighbour_size = 10000
    tabu_list = np.zeros((n+1, n+1))
    L = 40
    it = 0
    
    while True:
        print(cur_distance)
        it += 1
        
        # Check timeout
        time_now = round(time.time())
        time_diff = time_now - start_time
        if (seconds_timeout != -1 and time_diff > seconds_timeout):
            print("TIMEOUT")
            break

        # Ranomdely select some points
        sample_points = random.sample(range(1,n), min(300, n))
        neighbours = [(i, j) for i in sample_points for j in range(i+1, min(i+200, len(route))) if j - i > 1 and 
                      tabu_list[i, j] <= it]
        print(len(neighbours))
        if len(neighbours) == 0:
            # reset
            tabu_list = np.zeros((n+1, n+1))
            it = 1
            neighbours = [(i, j) for i in range(1, n) for j in range(i+1, len(route)) if j - i > 1 and 
                      tabu_list[i, j] <= it]
        
        if len(neighbours) > max_neighbour_size:
            neighbours = random.sample(neighbours, max_neighbour_size)
        routes = list(map(lambda x : two_opt_swap(x[0], x[1], route, n), neighbours))    
        distances = list(map(lambda x : compute_tour_distance_2(x, dist_matrix), routes))
        # Update tabu_list
        for x in neighbours:
            tabu_list[x[0], x[1]] = tabu_list[x[0], x[1]] + L
        
        y = np.argmin(distances)
        best_move = routes[y]
        best_distance = distances[y]

        if (best_distance < cur_distance):
            cur_distance = best_distance
            route = best_move
        else:
            break
            
    return route

def flatmap(f, items):
    return chain.from_iterable(map(f, items))

def three_opt_neighbours_random(route, sample_size):
    n = len(route)
    if (n * n/2 * n/4 < sample_size): # Very approximate approach to find out whether we should do full enumeration
        return three_opt_neighbours(route)
    else:
        return list(set(list(map(lambda x : tuple(sorted(random.sample(range(n+1), 3))), range(0, sample_size)))))

def three_opt_neighbours(route):
    n = len(route)
    return [(i, j, k) for i in range(0, n+1) for j in range(i+1, n+1) for k in range(j+1, n+1)]

def three_opt_swap(route, move):
    p = route
    a, c, e = move
    # without loss of generality, sort
    a, c, e = sorted([a, c, e])
    b, d, f = a+1, c+1, e+1

    return [p[:a+1] + p[b:c+1]    + p[e:d-1:-1] + p[f:], # 2-opt
    p[:a+1] + p[c:b-1:-1] + p[d:e+1]    + p[f:], # 2-opt
    p[:a+1] + p[c:b-1:-1] + p[e:d-1:-1] + p[f:], # 3-opt
    p[:a+1] + p[d:e+1]    + p[b:c+1]    + p[f:], # 3-opt
    p[:a+1] + p[d:e+1]    + p[c:b-1:-1] + p[f:], # 3-opt
    p[:a+1] + p[e:d-1:-1] + p[b:c+1]    + p[f:], # 3-opt
    p[:a+1] + p[e:d-1:-1] + p[c:b-1:-1] + p[f:]] # 2-opt

def three_opt_full(vertices, n, points, dist_matrix, seconds_timeout = 180):
        
    route = nearest_neighbour_improved(vertices, n, points)
    cur_distance = compute_tour_distance(route, dist_matrix)
    start_time = round(time.time())
    max_neighbour_size = 10000
    
    while True:
        print(cur_distance)
        
        # Check timeout
        time_now = round(time.time())
        time_diff = time_now - start_time
        if (seconds_timeout != -1 and time_diff > seconds_timeout):
            print("TIMEOUT")
            break

        neighbours = three_opt_neighbours_random(route, max_neighbour_size)
        routes = list(flatmap(lambda x : three_opt_swap(route, x), neighbours))
        distances = list(map(lambda x : compute_tour_distance(x, dist_matrix), routes))

        y = np.argmin(distances)
        best_move = routes[y]
        best_distance = distances[y]

        if (best_distance < cur_distance):
            cur_distance = best_distance
            route = best_move
        else:
            break
            
    return route

def iterated_local_search_3opt(vertices, n, points, dist_matrix, max_searches = 3, seconds_timeout = 180):
    
    cur_route = nearest_neighbour_improved(vertices, n, points)
    cur_distance = compute_tour_distance(cur_route, dist_matrix)

    best_route = cur_route
    best_distance = cur_distance
    
    for i in range(0, max_searches):
    
        # Start at new random location each time 2opt is run
        cur_route =  three_opt_full(vertices, n, points, dist_matrix, seconds_timeout)
        cur_distance = compute_tour_distance(cur_route, dist_matrix)

        if (cur_distance < best_distance):
            best_distance = cur_distance
            best_route = cur_route        
    
    return best_route

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    data_file = StringIO(input_data)
    data = pd.read_csv(data_file, sep=" ", names=["x", "y"], dtype={"x":float, "y":float})

    n = int(data["x"][0])
    print(n)
    vertices = []
    points = np.zeros((n, 2))
    for i in range(1, n+1):
        vertices.append(Vertex(i-1, data["x"][i], data["y"][i]))
        points[i-1,:] = np.array([data["x"][i], data["y"][i]])

    if n <= 10000:
        dist_matrix = compute_distance_matrix(vertices, n)
        solution = iterated_local_search_3opt(vertices, n, points, dist_matrix, max_searches = round(10000 / ( n* 8)), seconds_timeout = 1200)
    else:
        dist_matrix = np.zeros((n, n))
        solution = tabu_search2(vertices, n, points, dist_matrix, 6000)
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

