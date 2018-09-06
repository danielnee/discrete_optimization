#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from ortools.sat.python import cp_model

from io import StringIO

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
  """Print intermediate solutions."""

  def __init__(self, variables):
    self.__variables = variables
    self.__solution_count = 0
    self.__solutions = []

  def NewSolution(self):
    self.__solution_count += 1
    self.__solutions.append([self.Value(v) for v in self.__variables])

  def SolutionCount(self):
    return self.__solution_count

  def GetSolutions(self):
    return self.__solutions

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    data_file = StringIO(input_data)
    data = pd.read_csv(data_file, sep=" ", names=["vertices", "edges"])

    n = data.vertices[0]
    e = data.edges[0]
    edge_list = np.array(data[1:])

    print(n)

    degrees = np.zeros((n,1))
    for edge in edge_list:
        e1 = edge[0]
        e2 = edge[1]
        degrees[e1] += 1
        degrees[e2] += 1
    max_degree_node = np.argmax(degrees)

    max_colours = 100
    min_colours = 5
    MAX_TIME = 60.0

    for num_colours in range(min_colours, max_colours + 1):

        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        # Sets a time limit of 10 seconds.
        solver.parameters.max_time_in_seconds = MAX_TIME

        # Create the variables
        C = []
        for i in range(0, n):
            C.append(model.NewIntVar(0, num_colours - 1, "c_" + str(i)))

        # Create the constraints
        for edge in edge_list:
            e1 = edge[0]
            e2 = edge[1]
            model.Add(C[e1] != C[e2])

        # symmetry breaking
        # Chose vertex with highest degreee and assign it the first colour
        model.Add(C[max_degree_node] == 0)
        
        for i in range(1, num_colours):
            model.Add(C[i] <= i+1);

        model.Minimize(max(C))

        # Call the solver.
        solution_printer = SolutionPrinter(C)
        status = solver.SolveWithSolutionObserver(model, solution_printer)
        #print("Number of colours: %i" % num_colours)
        print('Number of solutions found: %i' % solution_printer.SolutionCount())
        
        if solution_printer.SolutionCount() > 0:
            break
            
    # Now ensure you get solution with minimum
    solutions = solution_printer.GetSolutions()
    max_color = [max(C) for C in solutions]
    best_index = np.argmin(max_color)
    node_count = max_color[best_index]
    solution = solutions[best_index]
    
    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

