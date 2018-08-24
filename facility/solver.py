#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from pulp import *
from io import StringIO

def euclidean_distance(x_1, x_2, y_1, y_2):
    return np.sqrt( np.square(x_1 - x_2)  + np.square(y_1 - y_2) )


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
    for j in C:
        prob += lpSum([xij[i,j] for i in F]) == 1,"%s must be assigned to at least one facility" % (j)

    ## Set constraint 2
    for i in F:
        prob += lpSum([xij[i,j] * c[j] for j in C]) <= f[i],"%s demand must not exceed capacity" % (i)

    ## Set constraint 3
    for i in F:
        for j in C:
            prob += xij[i,j] <= pi[i]

    import time

    start = time.time()
    ## Solve model
    prob.solve(solvers.PULP_CBC_CMD(maxSeconds=1200, threads=3,fracGap=0.0001))
    end = time.time()
    print(pulp.LpStatus[prob.status], "solution is: ", float(end - start), "sec")

    ## Print variables with value greater than 0 
    for v in prob.variables():
        if v.varValue>0:
            print(v.name, "=", v.varValue)

    # Print The optimal objective function value
    print("Total Cost = ", pulp.value(prob.objective))

    ## Print variables with value greater than 0 
    customer_assignments = {}
    for v in prob.variables():
        if v.varValue > 0:
            var_split = v.name.split("_")
            if (var_split[0] == "x"):
                customer_assignments[var_split[2]] = var_split[1]

    settings = [customer_assignments[str(i)] for i in range(0,m)]

    is_opt = str(1) if pulp.LpStatus[prob.status] == "Optimal" else str(0)

    output_data = str(pulp.value(prob.objective)) + ' ' + is_opt + '\n'
    output_data += ' '.join(map(str, settings))

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

