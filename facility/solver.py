#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from pulp import *
from io import StringIO

def euclidean_distance(x_1, x_2, y_1, y_2):
    return np.sqrt( np.square(x_1 - x_2)  + np.square(y_1 - y_2) )

def mip_solver(F, C, f, c, s, d, m, n):
    ### Create Problem/ Solver
    prob = LpProblem("The facility location problem", LpMinimize)

    ### Create decision variables
    xij = {}
    for i in F:
        for j in C:
            xij[i,j] = LpVariable("x_%s_%s" % (i,j), 0, 1, LpBinary)

    pi = {}
    for i in F:
        pi[i] = LpVariable("p_%s" % (i), 0, 1, LpBinary)

    ## Set the objective function 
    prob += lpSum([d[(i,j)]*xij[i,j] for i in F for j in C]) + lpSum([s[i] * pi[i] for i in F])
    
    ## Set constraint 1
    # Using great than or equal is significantly faster
    for j in C:
        prob += lpSum([xij[i,j] for i in F]) == 1,"%s must be assigned to at least one facility" % (j)
        
    ## Set constraint 2
    for i in F:
        prob += lpSum([xij[i,j] * c[j] for j in C]) <= f[i],"%s demand must not exceed capacity" % (i)
        
    ## Set constraint 3
    #bigM=m
    #for i in F:
    #    prob += lpSum([xij[i,j] for j in C]) <= bigM * pi[i],"%s bigM constraint" % (i)

    for i in F:
        for j in C:
            prob += xij[i,j] <= pi[i]
            
    import time

    start = time.time()
    ## Solve model
    prob.solve(solvers.PULP_CBC_CMD(maxSeconds=1800, threads=3,fracGap=0.0001))
    end = time.time()
    print(pulp.LpStatus[prob.status], "solution is: ", float(end - start), "sec")
    
    ## Print variables with value greater than 0 
    for v in prob.variables():
        if v.varValue>0:
            print(v.name, "=", v.varValue)

    # Print The optimal objective function value
    print("Total Cost = ", pulp.value(prob.objective))
    
    customer_assignments = dict.fromkeys(C, list())
    for v in prob.variables():
        if v.varValue > 0:
            var_split = v.name.split("_")
            if (var_split[0] == "x"):
                customer = int(var_split[2])
                facility = var_split[1]
                var_value = v.varValue
                
                temp = customer_assignments[customer].copy()
                temp.append((facility, var_value))
                customer_assignments[customer] = temp
    
    # Create the final solution
    used_capacity = dict.fromkeys(F, 0)
    solution = []
    for j in C:
        potential_facilities = list(map(lambda x: int(x[0]), customer_assignments[j]))
        potential_values = list(map(lambda x: x[1], customer_assignments[j]))
        sorted_values = np.argsort(potential_values)[::-1]
        allocated_customer = False
        for k in sorted_values:
            facility = potential_facilities[k]
            if (used_capacity[facility] + c[j]) <= f[facility]:
                used_capacity[facility] = used_capacity[facility] + c[j]
                solution.append(facility)
                allocated_customer = True
                break
    
        if (not allocated_customer):
            # Find a facility to allocate to
            sorted_distances = np.argsort(d[:,j])
            for k in sorted_distances:
                facility = F[k]
                if (used_capacity[facility] > 0) and ((used_capacity[facility] + c[j]) <= f[facility]):
                    used_capacity[facility] = used_capacity[facility] +  c[j]
                    solution.append(facility)
                    break

    print(len(solution))
    print(len(C))
    obj_value = objective_function(solution, d, s, C)
    is_valid = check_valid(solution, F, C, c, f)
    
    assert(is_valid)
    
    # prepare the solution in the specified output format
    is_opt = str(1) if pulp.LpStatus[prob.status] == "Optimal" else str(0)

    return solution, is_opt, obj_value

def check_valid(solution, F, C, c, f):
    # Check all customers have a facility
    assigned_all_customers = len(solution) == len(C)
    
    # Check capacity
    facilities_used = list(set(solution))
    solution_dict = {}
    for i in C:
        solution_dict[i] = solution[i]
    by_facility = reverse_dict(solution_dict)
    valid_count = 0
    for i in facilities_used:
        cur_customers = by_facility[i]
        valid_count += int(sum(map(lambda x : c[x], cur_customers)) <= f[i])
    capacity_valid = valid_count == len(facilities_used)
    return assigned_all_customers and capacity_valid

def reverse_dict(dictionary):
    newdict = {}
    for k, v in dictionary.items():
        newdict.setdefault(v, []).append(k)
    return newdict
   
def objective_function(solution, d, s, C):
    facilities_used = list(set(solution))
    return sum([d[solution[j], j] for j in C]) + sum(map(lambda i: s[i], facilities_used))

def generate_output(solution, optimal, objective):
    output_data = str(objective) + ' ' + str(optimal) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data

def greedy(F, C, f, c, s, d, m, n):
    
    costs_sorted = np.argsort(s)
    best_value = np.inf
    best_solution = None
    
    for i in F:
        available_facilities = list(costs_sorted[0:i+1])
        available_distances = d[available_facilities,]

        used_capacity = dict.fromkeys(available_facilities, 0)
        solution = []
        for j in C:
            sorted_distances = np.argsort(available_distances[:,j])
            for k in sorted_distances:
                facility = available_facilities[k]
                if (used_capacity[facility] + c[j]) <= f[facility]:
                    used_capacity[facility] = used_capacity[facility] +  c[j]
                    solution.append(facility)
                    break

        if len(solution) < len(C):
            continue
        
        cur_value = objective_function(solution, d, s, C)
        is_valid = check_valid(solution, F, C, c, f)
        
        if is_valid and (cur_value < best_value):
            best_value = cur_value
            best_solution = solution
        
    return best_solution, 0, best_value


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    data_file = StringIO(input_data)
    data = pd.read_csv(data_file, sep=" ", names=["a", "b", "c", "d"], dtype={"a":float, "b":float, "c":float, "d":float})

    n = int(data["a"][0])
    m = int(data["b"][0])
    print("n=%s" % n)
    print("m=%s" % m)

    facilities = data[1:(n+1)].rename({"a":"cost", "b":"capacity", "c":"x", "d":"y"}, axis="columns").reset_index(drop=True)
    customers = data[(n+1):].drop("d", axis=1).rename({"a":"demand", "b":"x", "c":"y"}, axis="columns").reset_index(drop=True)

    F = list(range(0, n))
    C = list(range(0, m))
    f = np.array(facilities.capacity)
    c = np.array(customers.demand)
    s = np.array(facilities.cost)
    d = np.zeros([n, m])

    all_pairs = [(i, j) for i in F for j in C]
    facility_locations = np.array(facilities[["x", "y"]])
    customer_locations = np.array(customers[["x", "y"]])

    for pair in all_pairs:
        i = pair[0]
        j = pair[1]
        d[i,j] = euclidean_distance(facility_locations[i,0], customer_locations[j,0], facility_locations[i,1], customer_locations[j,1])
    
    if n * m < 4000000:
        solution, optimal, objective = mip_solver(F, C, f, c, s, d, m, n)
    else:
        solution, optimal, objective = greedy(F, C, f, c, s, d, m, n)

    output_data = generate_output(solution, optimal, objective)

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

