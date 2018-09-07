#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import time

from io import StringIO

class Vertex:
    
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y
        
def euclidean_distance(vertex_1, vertex_2):
    return np.sqrt( np.square(vertex_1.x - vertex_2.x)  + np.square(vertex_1.y - vertex_2.y) )
        
def compute_distance_matrix(vertices, n):
    all_pairs = [(i, j) for i in range(0, n) for j in range(0, n) if i != j]
    dist_matrix = np.zeros((n, n))
    for k in all_pairs:
        x = k[0]; y = k[1]
        dist_matrix[x, y] = euclidean_distance(vertices[x], vertices[y])
    return dist_matrix    

def tour_list(routes):
    output = []
    for k in routes:
        output.append([i.index for i in k])
    
    return output

def compute_tour_distance(route, dist_matrix):
    head_route = route[:-1]
    tail_route = route[1:]
    pair_route = zip(head_route, tail_route)    
    return sum(map(lambda pair: dist_matrix[pair[0].index, pair[1].index], pair_route))

def compute_routes_distance(routes, dist_matrix):
    return sum([compute_tour_distance(route, dist_matrix) for route in routes])
    
def current_route_demand(route, customer_demands):
    return sum(map(lambda x: customer_demands[x.index], route))

def best_insertion_initialisation(customer_locations, customer_distances, customer_demands, n, v, c):
    # Initialise all routes
    depot = customer_locations[0]
    routes = []
    used_capacity = []
    for i in range(0, v):
        routes.append([depot, depot])
        used_capacity.append(0)

    # Add in decreasing order of demand
    order_to_add = list(reversed(np.argsort(customer_demands)))[:-1]
        
    return best_insertion(order_to_add, used_capacity, routes, customer_locations, 
                          customer_distances, customer_demands, n, v, c)

def best_insertion(customer_list, used_capacity, routes, customer_locations, 
                   customer_distances, customer_demands, n, v, c):
    routes = routes.copy()
    for customer in customer_list:
        # Iterate through all vans and insertions
        potential_locations = [(i, j) for i in range(0, v) for j in range(0, len(routes[i]) - 1)]

        best_route = None
        best_distance = np.inf
        best_demand = None
        best_van = None
        for pair in potential_locations:
            van = pair[0]
            insert_after = pair[1]

            route = routes[van]

            route = route[0:insert_after+1] + [customer_locations[customer]] + route[insert_after+1:]
            routes_copy = routes.copy()
            routes_copy[van] = route

            new_distance = compute_routes_distance(routes_copy, customer_distances)

            # Check capacity
            new_used_capacity = used_capacity[van] + customer_demands[customer]

            if new_used_capacity <= c and new_distance < best_distance:
                best_route = routes_copy
                best_distance = new_distance
                best_demand = customer_demands[customer]
                best_van = van

        if best_route is None:
            raise Exception("Couldn't add customer " + str(customer))

        used_capacity[best_van] = used_capacity[best_van] + best_demand
        routes = best_route
        
    return routes

def best_insertion_iteration(routes, customers_to_add, customer_locations, 
                   customer_distances, customer_demands, n, v, c):
    # Check current capacities
    used_capacity = []
    for i in range(0, v):
        route = routes[i]
        used_capacity.append(current_route_demand(route, customer_demands))
        
    # Add in decreasing order of demand
    demands = [customer_demands[i.index] for i in customers_to_add]
    order_to_add = list(reversed(np.argsort(demands)))
    customers = [customers_to_add[i].index for i in order_to_add]
    
    return best_insertion(customers, used_capacity, routes, customer_locations, 
                   customer_distances, customer_demands, n, v, c)

def random_van_removal(routes, customer_locations, v, max_vans):
    no_vans_to_remove = np.random.randint(1, max_vans + 1)
    van_lists = list(range(0, v))
    vans_to_remove = random.sample(van_lists, no_vans_to_remove)
    depot = customer_locations[0]
    
    routes_removed = [routes[i] for i in vans_to_remove]
    customers_removed =  [customer for route in routes_removed for customer in route]
    customers_removed = list(filter(lambda x : x.index != 0, customers_removed))
    
    routes_copy = routes.copy()
    for i in vans_to_remove:
        routes_copy[i] = [depot, depot]
        
    return (routes, customers_removed)    

def random_customer_removal(routes, customer_locations, n, removal_percent):
    no_customers_remove = int(np.floor(removal_percent * n))
    customer_list = list(range(1, n))
    customers_to_remove = random.sample(customer_list, no_customers_remove)
    removal_set = set(customers_to_remove)
    
    output_routes = []
    for route in routes:
        
        output_route = []
        cur_index = 0
        for i in range(1, len(route)):
            cur_customer = route[i].index
            if cur_customer in removal_set:
                output_route.extend(route[cur_index:i].copy())
                cur_index = i + 1
        
        output_route.extend(route[cur_index:].copy())
        output_routes.append(output_route)
        
    customers_to_remove = [customer_locations[i] for i in customers_to_remove] 
       
    return (output_routes, customers_to_remove)       

def local_search(customer_locations, customer_distances, customer_demands, n, v, c, seconds_timeout = 300):
    
    routes = best_insertion_initialisation(customer_locations, customer_distances, customer_demands, n, v, c)
    best_distance = compute_routes_distance(routes, customer_distances)
    
    start_time = round(time.time())
    it = 0
    
    print(best_distance) 

    while True:
        it += 1
        
        if it % 100 == 0:
            print(best_distance)        
        
        # Check timeout
        time_now = round(time.time())
        time_diff = time_now - start_time
        if (seconds_timeout != -1 and time_diff > seconds_timeout):
            print("TIMEOUT")
            break
            
        # Random customer removal
        new_routes, customers_removed = random_customer_removal(routes, customer_locations, n, 0.2)
        try:
            new_routes = best_insertion_iteration(new_routes, customers_removed, customer_locations, 
                   customer_distances, customer_demands, n, v, c)
        except:
            new_routes = routes
        new_distance = compute_routes_distance(new_routes, customer_distances)
        if (new_distance < best_distance):
            routes = new_routes            
            best_distance = new_distance
            
        # Random van removal    
        max_vans = np.floor(v / 2)
        new_routes, customers_removed = random_van_removal(routes, customer_locations, v, max_vans)
        try:
            new_routes = best_insertion_iteration(new_routes, customers_removed, customer_locations, 
                   customer_distances, customer_demands, n, v, c)
        except:
            new_routes = routes
        new_distance = compute_routes_distance(new_routes, customer_distances)
        if (new_distance < best_distance):
            routes = new_routes            
            best_distance = new_distance
            
        # Repeat until timeout reached    
        
    return routes

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

   # parse the input
    data_file = StringIO(input_data)
    data = pd.read_csv(data_file, sep=" ", names=["a", "b", "c"])

    n = int(data.a[0])
    v = int(data.b[0])
    c = int(data.c[0])

    print(n)
    print(v)

    customer_demands = [None] * n
    customer_locations = [None] * n
    for i in range(0, n):
        index = i + 1
        customer_demands[i] = int(data.a[index])
        customer_locations[i] = Vertex(i, data.b[index], data.c[index])

    # Pre-compute customer-customer distances
    customer_distances = compute_distance_matrix(customer_locations, n)
    routes = local_search(customer_locations, customer_distances, customer_demands, n, v, c, 30)

    obj = compute_routes_distance(routes, customer_distances)
    tour = tour_list(routes)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for route in tour:
         outputData += ' '.join([str(i) for i in route]) + "\n"

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

