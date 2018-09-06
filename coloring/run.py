import pandas as pd
import numpy as np

from ortools.sat.python import cp_model

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
  """Print intermediate solutions."""

  def __init__(self, variables):
    self.__variables = variables
    self.__solution_count = 0
    self.__solutions = []

  def NewSolution(self):
    self.__solution_count += 1
    self.__solutions.append([self.Value(v) for v in self.__variables])
    
    for v in self.__variables:
      print('%s = %i' % (v, self.Value(v)), end = ' ')
    print()

  def SolutionCount(self):
    return self.__solution_count

  def GetSolutions(self):
    return self.__solutions

data_file = "data/gc_20_1"
data = pd.read_csv(data_file, sep=" ", names=["vertices", "edges"])

n = data.vertices[0]
e = data.edges[0]
edge_list = np.array(data[1:])

max_colours = 2
model = cp_model.CpModel()
solver = cp_model.CpSolver()

# Sets a time limit of 10 seconds.
#solver.parameters.max_time_in_seconds = 300.0

# Create the variables
C = []
for i in range(0, n):
    C.append(model.NewIntVar(0, max_colours - 1, "c_" + str(i)))
    
# Create the constraints
for edge in edge_list:
    e1 = edge[0]
    e2 = edge[1]
    model.Add(C[e1] != C[e2])

# Call the solver.
solution_printer = SolutionPrinter(C)
status = solver.SearchForAllSolutions(model, solution_printer)
print('\nNumber of solutions found: %i' % solution_printer.SolutionCount())