#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time

from queue import PriorityQueue
from itertools import compress
from copy import deepcopy
from collections import deque
from io import StringIO

def objective_func(items, values):
    return sum(compress(values, items))

def current_weight(items, weights):
    return sum(compress(weights, items))

def valid_solution(items, weights, K):
    return current_weight(items, weights) <= K

def greedy_by_value(weights, values, n, K, descending=True):
    solution = n * [0]
    values_descending_index = reversed(np.argsort(values)) if descending else np.argsort(values)
    for i in values_descending_index:
        if (current_weight(solution, weights) + weights[i]) <= K:
            solution[i] = 1
            
    return (objective_func(solution, values), solution)

def no_constraint_relaxation(values, items):
    return objective_func(items, values)

def linear_relaxation(values, weights, items, K):
    value_weight_ratio = np.array(values) / np.array(weights)
    order_value_weight = list(reversed(np.argsort(values)))
    cur_value = 0
    cur_weight = 0
    for i in range(0, len(order_value_weight)):
        probe_index = order_value_weight[i]
        probe_weight = weights[probe_index]
        probe_value = values[probe_index]
        
        if items[probe_index] == 0:
            continue
        
        if probe_weight >= K:
            continue
        
        new_weight = cur_weight + probe_weight
        if new_weight <= K:
            cur_weight += probe_weight
            cur_value += probe_value
        else:
            # Add the proportion of the last item
            remaining = K - cur_weight
            proportion = remaining / probe_weight
            cur_value += proportion * probe_value
            cur_weight += proportion * probe_weight
            break
                 
    assert(cur_weight <= K)
    return cur_value

# Might be multiple optimal but ignore for now
# weight, K, values
def branch_and_bound_breadth_first_recr(weights, values, n, K, i, solution, cur_optimisitic_value, cur_capacity, cur_value, seconds_timeout = 60):
    global best_solution
    global best_solution_objective_value 
    global start_time
    
    # Check timeout
    time_now = round(time.time())
    time_diff = time_now - start_time
    if (seconds_timeout != -1 and time_diff > seconds_timeout):
        print("TIMEOUT")
        return
    
    # Base cases
    
    # Current solution not feasible
    if not(valid_solution(solution, weights, K)):
        return
    # Current solutions optimistic evaluation is worse than best solution found
    elif cur_optimisitic_value < best_solution_objective_value:
        return
    # Best solution found - could be multiple
    elif cur_optimisitic_value == cur_value:
        best_solution_objective_value = cur_value
        best_solution = deepcopy(solution)
        print(best_solution_objective_value)
        return
        
    # Go left down the tree
    solution[i] = 1
    temp_value = cur_value + values[i]
    temp_capacity = cur_capacity + weights[i]
    j = i + 1
    branch_and_bound_breadth_first_recr(weights, values, n, K, j, solution, cur_optimisitic_value, temp_capacity, temp_value, seconds_timeout)
    
    # Go right down the tree
    solution[i] = 0
    optimistic_solution = deepcopy(solution)
    optimistic_solution[i+1:n] = [1] * (n-i-1)
    temp_optimisitic_value = linear_relaxation(values, weights, optimistic_solution, K)
    j = i + 1
    branch_and_bound_breadth_first_recr(weights, values, n, K, j, solution, temp_optimisitic_value, cur_capacity, cur_value, seconds_timeout)
    
    return

def branch_and_bound_breadth_first(weights, values, n, K):
    global best_solution
    global best_solution_objective_value 
    global start_time
    
    full_solution = [1] * n
    optimistic_eval = linear_relaxation(values, weights, full_solution, K)
    best_solution = [0] * n
    best_solution_objective_value = 0
    initial_solution = [0] * n
    start_time = round(time.time())
    branch_and_bound_breadth_first_recr(weights, values, n, K, 0, initial_solution, optimistic_eval, 0, 0, 60)
    
    return (best_solution_objective_value, best_solution)

def branch_and_bound_best_first(weights, values, n, K, seconds_timeout = 60):

    class Solution(object):

        def __init__(self, value, capacity, solution, node, optimistic_eval):
            self.value = value
            self.capacity = capacity
            self.solution = deepcopy(solution)
            self.node = node
            self.optimistic_eval = optimistic_eval

        def __lt__(self, other):
            # Do the opposite as we want it reversed
            return self.optimistic_eval >= other.optimistic_eval


    full_solution = [1] * n
    optimistic_eval = linear_relaxation(values, weights, full_solution, K)
    best_solution = [0] * n
    best_solution_objective_value = 0
    solution_queue = PriorityQueue()
    start_time = round(time.time())

    solution = [0] * n
    cur_value = 0
    cur_capacity = 0
    i = 0
    while True:
        # Check timeout
        time_now = round(time.time())
        time_diff = time_now - start_time
        if (seconds_timeout != -1 and time_diff > seconds_timeout):
            print("TIMEOUT")
            break
        
        # Only move down the tree if not a root
        if i < (n-1):
            # Go left down the tree
            solution[i] = 1
            temp_value = cur_value + values[i]
            temp_capacity = cur_capacity + weights[i]
            j = i + 1
            # Store temp_value, temp_capacity, solution, j, optimistic_eval
            # Only store if valid
            if temp_capacity <= K:
                solution_queue.put(Solution(temp_value, temp_capacity, solution, j, optimistic_eval))

            # Go right down the tree
            solution[i] = 0
            # Check
            optimistic_solution = deepcopy(solution)
            optimistic_solution[i+1:n] = [1] * (n-i-1)
            temp_optimisitic_value = linear_relaxation(values, weights, optimistic_solution, K)
            j = i + 1
            # Store cur_value, cur_capacity, solution, j, temp_optimisitic_value
            # Only store if valid
            if cur_capacity <= K:
                solution_queue.put(Solution(cur_value, cur_capacity, solution, j, temp_optimisitic_value))
        elif i == (n-1):
            # At a root node
            if cur_value > best_solution_objective_value:
                print(cur_value)
                best_solution_objective_value = cur_value
                best_solution = deepcopy(solution)

        # Check if empty
        if solution_queue.empty():
            break

        # Choose best
        cur_best = solution_queue.get()

        # if worse that current best solution stop
        if cur_best.optimistic_eval < best_solution_objective_value:
            break

        # else setup next solution
        cur_value = cur_best.value
        cur_capacity = cur_best.capacity
        solution = cur_best.solution
        i = cur_best.node
        optimistic_eval = cur_best.optimistic_eval
        
    return (best_solution_objective_value, best_solution)

def branch_and_bound_lds_recr(weights, values, n, K, i, wave, solution, cur_optimisitic_value, cur_capacity, 
                              cur_value, seconds_timeout = 60):
    global best_solution
    global best_solution_objective_value 
    global start_time
    
    # Check timeout
    time_now = round(time.time())
    time_diff = time_now - start_time
    if (seconds_timeout != -1 and time_diff > seconds_timeout):
        print("TIMEOUT")
        return
    
    # Base cases
    
    # Current solution not feasible
    if not(valid_solution(solution, weights, K)):
        return
    # Current solutions optimistic evaluation is worse than best solution found
    elif cur_optimisitic_value < best_solution_objective_value:
        return
    # Best solution found - could be multiple
    elif cur_optimisitic_value == cur_value:
        best_solution_objective_value = cur_value
        best_solution = deepcopy(solution)
        print(best_solution_objective_value)
        return
        
    if wave == 0:
        # Go Left
        solution[i] = 1
        temp_value = cur_value + values[i]
        temp_capacity = cur_capacity + weights[i]
        j = i + 1
        branch_and_bound_lds_recr(weights, values, n, K, j, wave, solution, cur_optimisitic_value, temp_capacity, 
                                  temp_value, seconds_timeout)
    else:
        # Go right
        solution[i] = 0
        optimistic_solution = deepcopy(solution)
        optimistic_solution[i+1:n] = [1] * (n-i-1)
        temp_optimisitic_value = linear_relaxation(values, weights, optimistic_solution, K)
        j = i + 1
        branch_and_bound_lds_recr(weights, values, n, K, j, wave - 1, solution, 
                                            temp_optimisitic_value, cur_capacity, cur_value, seconds_timeout)
    
        # Now go left with current wave
        solution[i] = 1
        temp_value = cur_value + values[i]
        temp_capacity = cur_capacity + weights[i]
        j = i + 1
        branch_and_bound_lds_recr(weights, values, n, K, j, wave, solution, cur_optimisitic_value, temp_capacity, 
                                  temp_value, seconds_timeout)
    
    return

def branch_and_bound_lds(weights, values, n, K, seconds_timeout = 60):
    global best_solution
    global best_solution_objective_value 
    global start_time
    
    # Outer loop for waves
    full_solution = [1] * n
    optimistic_eval = linear_relaxation(values, weights, full_solution, K)
    best_solution = [0] * n
    best_solution_objective_value = 0
    initial_solution = [0] * n
    start_time = round(time.time())

    for i in range(0, n+1):
        branch_and_bound_lds_recr(weights, values, n, K, 0, i, deepcopy(initial_solution), optimistic_eval, 0, 0, 60)
        
    return (best_solution_objective_value, best_solution)    

def dynamic_programming(weights, values, n, K):
    # Compute dynamic programming matrix
    T = np.zeros((n+1, K+1))

    for j in range(0, n+1):
        for k in range(0, K+1):
            if (j == 0) or (k == 0):
                T[j,k] = 0
            elif weights[j-1] <= k:
                T[j,k] = max(values[j-1] + T[j-1, k - weights[j-1]],  T[j-1, k])
            else:
                T[j,k] = T[j-1, k]

    T = np.transpose(T)
     
    # Produce traceback
    k = K
    result = deque()
    for j in range(n, 0, -1):
        if T[k,j] == T[k,j-1]:
            result.appendleft(0)
        else:
            result.appendleft(1)
            k -= weights[j-1]
    
    result = list(result)
    return (objective_func(result, values), result)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    data_file = StringIO(input_data)
    data = pd.read_csv(data_file, sep=" ", names=["values", "weights"])

    n = data["values"][0]
    K = data["weights"][0]
    values = n * [0]
    weights = n * [0]
    for i in range(1, n+1):
        values [i-1] = data["values"][i]
        weights[i-1] = data["weights"][i]

    # Greedy by value
    #a = greedy_by_value(weights, values, n, K)
    #print(a)
    #print(current_weight(a, weights))  

    # Greedy by value weight ratio
    #value_weight_ratio = np.array(values) / np.array(weights)
    #a = greedy_by_value(weights, value_weight_ratio, n, K)
    #print(a)
    #print(current_weight(a, weights))
    #print(objective_func(a[1], values))  
    #a = (objective_func(a[1], values), a[1])

    # Greedy by smallest weight
    #a = greedy_by_value(weights, weights, n, K, descending=False)
    #print(a)
    #print(current_weight(a, weights))

    # Causes stack overflow on problem 6
    #a = branch_and_bound_breadth_first(weights, values, n, K)


    #a = branch_and_bound_best_first(weights, values, n, K, seconds_timeout = 300)
 #   if a[0] == 0:
 #       value_weight_ratio = np.array(values) / np.array(weights)
 #       a = greedy_by_value(weights, value_weight_ratio, n, K)
 #       a = (objective_func(a[1], values), a[1])

    #a = branch_and_bound_lds(weights, values, n, K, seconds_timeout = 300)
    #if a[0] == 0:
    #    value_weight_ratio = np.array(values) / np.array(weights)
    #    a = greedy_by_value(weights, value_weight_ratio, n, K)
    #    a = (objective_func(a[1], values), a[1])

    if (n * K) <= 500000000:
        a = dynamic_programming(weights, values, n, K)
    else:
        value_weight_ratio = np.array(values) / np.array(weights)
        a = greedy_by_value(weights, value_weight_ratio, n, K)
        print(a)
        print(current_weight(a, weights))
        print(objective_func(a[1], values))  
        a = (objective_func(a[1], values), a[1])
    
    # prepare the solution in the specified output format
    output_data = str(a[0]) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, a[1]))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

