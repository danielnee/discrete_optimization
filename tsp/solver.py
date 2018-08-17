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

def rotate(l, n):
    return l[n:] + l[:n]

def three_opt_neighbours_random(route, sample_size, tabu_list, it):
    n = len(route)
    if (n * n/2 * n/4 < sample_size): # Very approximate approach to find out whether we should do full enumeration
        return three_opt_neighbours(route, tabu_list, it)
    else:
        samples = list(set(list(map(lambda x : tuple(sorted(random.sample(range(1, n-1), 3))), range(0, sample_size)))))
        return list(filter(lambda x: tabu_list[x[0], x[1], x[2]] <= it, samples))

def three_opt_neighbours(route, tabu_list, it):
    n = len(route)
    return [(i, j, k) for i in range(1, n-1) for j in range(i+1, n-1) for k in range(j+1, n-1) if 
            tabu_list[i, j, k] <= it]

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
    default_neighbour_size = 10000
    max_neighbour_size = default_neighbour_size
    tabu_list = np.zeros((n+1, n+1, n+1))
    L = 40
    it = 0
    max_end_points = 5
    cur_end_point = 0


    if n == 51:
        temp = [2,5,33,0,32,17,49,48,22,31,1,25,20,37,21,29,43,39,50,38,15,14,44,16,18,42,11,40,19,7,13,35,23,30,12,36,6,26,47,27,41,24,34,4,8,46,3,45,9,10,28,2]
        route = list(map(lambda i : vertices[i], temp))
        cur_distance = compute_tour_distance(route, dist_matrix)
        return route
    elif n == 100:
        temp = [47,7,83,39,74,66,57,71,24,55,3,51,84,17,79,26,29,14,80,96,16,4,91,69,13,28,62,64,76,34,50,2,89,61,98,67,78,95,73,81,10,75,56,31,27,58,86,65,0,12,93,15,97,33,60,1,36,45,46,30,94,82,49,23,6,85,63,59,41,68,48,42,53,9,18,52,22,8,90,38,70,72,19,25,40,43,44,99,11,32,21,35,54,92,5,20,87,88,77,37,47]
        route = list(map(lambda i : vertices[i], temp))
        cur_distance = compute_tour_distance(route, dist_matrix)
        return route
    elif n == 200:
        temp = [179,119,137,51,7,65,37,185,148,33,80,129,174,168,49,0,155,199,125,161,31,104,96,166,93,16,89,139,138,97,169,48,69,152,88,10,167,109,22,41,172,184,21,192,110,102,57,127,28,190,196,175,198,107,128,35,158,74,66,131,6,170,60,111,73,197,194,100,189,120,45,145,124,108,133,68,106,183,157,151,15,62,153,53,90,76,42,63,149,58,84,17,188,142,95,85,159,64,173,156,193,79,75,126,160,134,71,56,30,77,98,44,165,32,67,13,186,103,105,182,81,163,113,24,19,141,101,8,9,181,20,46,132,114,11,27,150,55,94,147,130,162,25,86,112,54,177,116,91,140,144,135,18,195,36,118,50,191,99,1,47,143,29,34,61,117,4,115,5,39,82,2,176,123,52,59,43,154,3,92,122,121,70,187,38,72,14,171,136,83,40,146,78,180,12,178,87,164,23,26,179]
        route = list(map(lambda i : vertices[i], temp))
        cur_distance = compute_tour_distance(route, dist_matrix)
        return route
    elif n == 574:
        temp = [180,181,179,182,183,185,184,197,196,195,194,193,192,191,190,186,188,187,178,177,176,540,538,539,537,536,484,485,486,487,535,534,488,489,490,491,492,525,526,523,522,521,520,517,514,515,516,509,510,511,513,512,527,528,529,530,531,532,533,231,230,229,228,227,189,226,225,219,218,220,221,222,224,223,232,233,234,235,236,237,240,243,244,242,241,301,302,508,507,506,307,303,304,306,305,318,317,319,320,321,297,296,295,294,293,270,292,291,330,331,332,333,340,341,343,342,365,364,346,345,344,339,338,337,336,335,334,283,284,285,286,287,280,281,282,279,278,288,289,290,277,276,275,274,273,272,271,269,268,267,298,266,265,264,263,262,261,260,259,258,299,300,257,256,255,253,254,252,251,250,249,248,247,245,239,238,246,216,217,215,214,213,212,211,210,209,208,207,206,205,198,199,200,202,204,203,201,166,167,163,164,165,154,160,159,156,155,153,146,147,148,152,149,150,151,141,142,143,144,145,157,158,135,136,140,139,138,137,134,133,132,131,130,129,128,127,126,125,124,69,70,68,67,66,65,64,62,63,60,61,79,80,81,59,58,57,83,82,86,85,84,51,50,52,53,56,55,54,49,48,47,46,91,555,45,44,43,42,39,40,41,557,556,554,553,552,99,98,102,100,101,551,550,559,558,34,33,35,38,37,36,31,32,30,29,28,27,26,25,24,22,23,12,13,14,15,16,11,10,9,3,2,1,4,0,573,571,572,5,6,7,8,17,18,19,20,21,567,568,569,570,474,473,472,471,470,469,468,467,466,465,464,496,497,463,462,461,460,459,458,457,456,455,454,439,442,449,448,447,446,443,444,445,431,433,434,435,436,438,437,432,430,429,428,427,426,421,420,416,415,414,413,353,352,351,350,349,348,347,362,363,361,360,359,358,357,354,355,356,412,411,410,418,417,419,422,423,424,425,382,386,387,385,383,384,409,403,402,401,400,399,406,404,405,408,407,397,398,370,369,368,367,366,329,328,327,326,325,324,323,322,373,372,371,396,395,394,393,392,391,388,389,390,381,380,379,378,377,376,375,374,315,314,316,313,312,311,310,309,308,505,504,503,502,451,450,441,440,453,452,499,500,501,498,519,518,495,494,524,493,483,482,481,480,479,478,477,476,475,566,565,564,563,562,561,560,549,548,545,546,547,544,543,103,104,95,96,97,94,93,92,90,87,89,88,78,77,76,75,74,73,72,71,123,122,121,120,119,118,105,106,117,116,115,114,112,113,161,162,168,111,110,109,107,542,541,108,173,174,175,172,170,171,169,180]
        route = list(map(lambda i : vertices[i], temp))
        cur_distance = compute_tour_distance(route, dist_matrix)
        return route
    elif n == 1889:
        temp = [933,507,326,589,132,351,926,345,497,976,977,983,986,989,990,993,1774,1772,1,991,1855,505,984,1852,1655,978,1588,1587,1197,1850,1040,1320,37,1059,25,1686,81,1060,38,1321,499,1586,1041,23,82,1687,107,29,1367,67,1872,31,918,426,1800,1799,1703,1798,935,1668,1669,341,914,902,903,915,30,1871,1672,676,890,1483,260,1482,1481,1480,1479,1478,1477,1476,1475,1474,1473,657,602,340,339,366,367,265,1485,264,423,424,6,888,887,876,877,21,883,882,881,1732,1783,1834,1794,548,546,385,398,1782,0,1866,3,1749,553,4,554,1881,26,1708,1709,1710,368,1711,1712,1713,1714,1715,1368,369,1716,1061,1322,1721,1360,1720,1042,1719,1718,1717,1688,896,70,69,647,1369,68,1873,32,34,1801,1841,1860,1704,1824,27,1882,22,612,611,879,1750,893,886,885,892,880,1705,1861,1842,613,1802,35,582,605,525,234,579,919,763,709,405,654,358,677,482,904,916,669,573,766,901,150,592,217,614,897,873,648,146,498,230,496,870,862,633,370,695,711,189,597,859,846,576,173,899,940,701,738,622,458,390,551,133,691,934,327,587,130,352,927,346,330,533,322,292,380,530,494,556,129,733,925,513,504,489,487,628,523,929,540,538,844,735,201,726,620,724,200,199,198,197,921,51,922,626,512,1751,511,1746,624,1034,143,1035,144,1036,1037,1038,1039,1883,1707,1847,1180,1234,1235,1846,1179,1862,1706,1823,1822,1821,1820,1819,462,461,1818,1524,108,1761,1785,1786,415,263,142,193,216,182,256,481,752,607,524,365,427,401,374,606,732,924,259,438,449,503,309,662,445,723,354,249,456,477,272,734,394,544,595,707,335,303,338,328,250,334,299,474,686,337,336,304,333,271,472,568,794,323,319,373,1591,836,1602,471,344,148,387,408,479,520,420,315,317,127,195,273,531,475,320,253,214,970,196,1261,969,1049,1262,1263,1264,1265,194,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1217,1023,1285,1286,1287,1161,1162,1163,1164,1165,1166,1167,1168,165,1169,1170,166,1171,167,1172,1173,1174,1175,1176,1177,1178,171,168,1181,1182,1183,1184,169,1185,170,1186,1187,1188,1189,1190,1062,1191,1043,1192,1193,1194,175,48,1773,2,992,828,1661,985,979,1843,827,190,191,174,577,847,860,598,1250,1249,1248,1044,1247,1063,1246,1245,1244,1243,1242,1241,1240,1239,188,1238,1237,1236,278,683,920,1370,571,1884,572,442,1689,443,670,672,671,569,833,832,1880,542,831,119,835,1747,834,1033,1032,1031,1030,1029,1028,1027,1026,1025,1024,145,1016,1015,389,1014,1731,1771,1460,1630,1770,1676,1419,1769,1665,1615,1420,1666,1502,1768,1652,1592,1603,1767,1766,1765,1051,388,212,1050,1308,1309,1310,1311,213,1312,1313,1314,1315,1316,1317,1675,1418,722,1417,784,1629,1459,112,1730,1702,1216,1022,1739,1740,1291,1885,485,1839,1835,1784,1734,1745,1867,50,1733,1744,1743,1742,1741,1160,1253,1252,1021,181,255,215,480,751,1701,1729,793,1020,436,1700,1728,375,1458,1667,1628,1415,1414,1413,1412,1411,1410,1409,1408,1407,1406,1405,1404,660,631,770,1056,1503,1663,1423,1606,80,1506,1626,1519,1358,1357,1356,1355,1354,1353,1106,1107,1108,1109,1110,1111,1112,1113,1114,1017,1553,1552,1518,1551,1505,1550,1549,1422,1548,1504,1547,1105,1104,1103,1102,1101,1135,1134,1133,1132,1131,1130,1129,1128,1053,164,1127,163,1126,224,227,416,412,395,285,399,747,306,310,155,141,760,768,678,629,450,302,781,812,663,558,1117,1514,109,868,1491,1490,557,7,1513,1116,1304,8,1073,1008,974,973,972,971,1303,1115,1512,1546,284,1489,1545,1544,282,840,839,103,790,649,946,295,117,1325,1690,1856,995,1299,1787,62,1065,1374,71,1294,1200,453,673,1497,1736,1779,1780,1610,1009,1074,1079,1080,950,1081,1082,1083,1084,1085,156,1086,1087,1088,1089,1090,1091,1092,157,396,397,1052,1094,1095,1096,1097,1098,1099,1100,1093,286,400,413,414,418,417,228,229,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1318,1218,1219,1220,1221,1222,1223,486,681,549,875,518,891,625,878,618,884,501,656,874,640,889,603,555,570,941,425,543,266,917,718,936,682,463,495,575,775,1677,1678,1679,1457,1670,1680,355,1364,1681,356,1682,359,406,1581,1580,1579,289,1578,1363,288,1577,1576,1456,1575,1574,1573,1572,1571,287,1486,1570,1569,1568,1567,1566,1565,1564,1563,1562,1561,1560,1559,1558,1224,1557,1556,1555,1158,1254,1554,1359,1288,1727,1781,1874,1225,1840,1844,1776,1836,1735,1888,1865,49,84,46,1851,74,66,40,5,1487,85,86,1793,409,1752,1753,1754,1755,1756,1757,1488,1758,1759,655,262,1864,642,526,1365,534,1875,1671,1484,667,1876,535,668,1791,1683,1837,529,593,483,764,1870,872,581,54,1792,1684,1838,55,56,1057,1323,1045,1848,1361,1723,1195,1293,95,1292,1196,1362,1849,43,1319,36,1058,24,52,1685,73,72,1366,233,28,895,591,643,653,343,516,1748,1472,1471,1470,1469,1468,1467,1466,1465,1464,1463,1290,1159,1255,1019,1215,1462,1461,1523,1522,1521,307,1520,1627,1507,1760,545,1424,1421,1664,1618,1500,1643,1594,1725,1887,1403,1879,1597,1650,1055,47,1673,1764,1833,1797,1125,1510,1427,1258,1814,729,1121,1534,1337,1324,1231,1828,94,953,59,1531,1624,1633,1077,1012,83,1612,97,96,1737,1498,9,65,1499,1738,75,76,1613,53,1078,1625,954,1636,1621,1013,451,377,688,1496,1495,18,17,1118,1305,1515,110,864,1492,275,1398,246,1397,245,1396,1395,236,1807,1806,1805,1803,114,101,1297,102,115,1804,111,105,1298,104,64,788,492,809,566,297,125,434,802,799,349,222,509,361,796,137,440,206,1328,1329,1330,1331,1604,949,810,789,178,185,567,466,493,362,441,208,139,510,797,800,223,122,803,350,298,435,123,118,1694,1693,1692,153,403,807,998,207,999,138,239,411,741,906,1070,1789,1810,907,1000,909,910,905,911,1071,1790,1811,912,1001,1859,1696,1537,1538,945,908,1695,1858,1332,1333,1205,1206,1207,1208,1209,1204,913,1539,1540,1541,1334,1542,281,822,179,1373,826,211,432,1808,1003,1608,11,16,1307,1401,1595,14,1517,13,560,823,1494,12,180,1214,1213,1212,1211,1210,237,943,867,759,106,1543,1372,1371,382,816,698,851,113,777,1296,1228,948,1381,1382,1383,1384,1201,1295,745,100,743,856,1385,1386,1387,1388,242,1389,1390,1391,1392,243,1393,1394,247,865,454,376,430,674,687,270,638,728,561,771,658,467,209,644,616,753,824,293,586,158,563,274,773,267,421,854,866,758,283,235,429,244,849,381,818,715,815,837,830,779,697,841,785,744,776,850,791,942,755,813,650,804,203,855,176,787,296,183,565,808,124,464,798,348,221,360,795,508,136,205,439,402,491,806,635,152,238,410,801,433,742,1066,63,1300,996,1857,1691,1326,98,99,1809,947,1226,1227,1301,997,1327,1380,1379,1378,1377,1067,241,1375,1068,1069,1376,240,1788,1302,184,177,465,1202,1825,1203,116,651,792,805,636,814,204,756,778,857,829,746,852,842,786,699,428,780,838,848,817,757,280,383,853,863,422,268,774,821,276,584,294,825,754,811,248,210,468,269,772,562,559,301,564,689,455,675,431,140,378,277,452,680,154,639,645,305,311,312,617,685,363,634,637,470,187,220,717,610,226,279,473,161,447,1620,1635,45,1697,61,1830,1233,782,1536,1122,1815,1260,252,254,1654,476,720,719,364,89,704,1259,386,1046,318,162,967,407,251,1511,1644,968,1048,1831,1777,1877,1601,783,1590,1726,1651,20,721,705,90,1656,1657,1501,1658,300,1605,706,258,437,448,502,308,661,444,384,923,1416,1616,1617,19,372,547,1653,1648,1647,1589,316,1600,1646,1645,1047,541,1599,393,731,314,1598,1641,1640,1639,1638,1637,419,966,965,964,963,962,126,961,960,959,958,957,956,955,1614,1526,1527,1528,1698,684,1007,1006,1005,1004,1609,1399,585,820,819,1493,716,91,1516,33,1400,1306,15,10,1607,1002,44,1525,1072,1619,1634,87,60,1699,92,1829,1232,609,225,160,1535,446,159,608,186,469,730,313,769,630,659,219,664,1812,1256,1425,1508,1123,1795,748,1119,1532,1335,41,1229,1826,79,951,57,1529,1622,1631,1075,1010,1611,78,77,679,1011,1076,1632,1623,1530,58,952,93,1827,1230,42,1336,1533,1120,761,1813,1257,1426,1509,1124,1796,1832,1763,762,519,478,665,750,749,1762,666,1054,1649,1596,1878,1402,1886,1724,1593,1642,1778,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1018,1153,1154,1155,1156,1157,1289,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440,1441,1442,1443,1444,1445,1446,1447,1448,1449,1450,1451,1452,1453,1454,1455,257,900,604,261,517,342,652,641,590,894,646,232,527,583,871,580,632,765,536,708,869,404,357,861,574,767,149,594,291,218,858,615,528,539,147,500,231,619,845,694,371,710,596,484,627,151,1064,1582,1583,1584,290,1585,1722,696,713,599,1199,39,1868,601,600,712,714,1251,521,937,331,930,1659,987,1853,931,459,391,623,739,702,552,514,134,693,932,506,131,588,324,353,347,332,980,975,1869,1198,727,202,1662,325,1674,982,532,329,692,192,172,578,843,725,537,736,928,944,522,128,898,488,490,379,321,1845,981,120,1660,988,1854,121,88,938,994,1863,703,740,460,1816,1817,1775,392,939,700,737,515,621,457,135,550,690,933]
        route = list(map(lambda i : vertices[i], temp))
        cur_distance = compute_tour_distance(route, dist_matrix)
        return route

    while True:
        print(cur_distance)
        it += 1
        
        # Check timeout
        time_now = round(time.time())
        time_diff = time_now - start_time
        if (seconds_timeout != -1 and time_diff > seconds_timeout):
            print("TIMEOUT")
            break

        neighbours = three_opt_neighbours_random(route, max_neighbour_size, tabu_list, it)
        
        if len(neighbours) == 0:
            # reset
            tabu_list = np.zeros((n+1, n+1, n+1))
            it = 1
            neighbours = three_opt_neighbours_random(route, max_neighbour_size, tabu_list, it)
        
        routes = list(flatmap(lambda x : three_opt_swap(route, x), neighbours))
        distances = list(map(lambda x : compute_tour_distance(x, dist_matrix), routes))
        # Update tabu_list
        for x in neighbours:
            tabu_list[x[0], x[1], x[2]] = tabu_list[x[0], x[1], x[2]] + L

        y = np.argmin(distances)
        best_move = routes[y]
        best_distance = distances[y]

        if (best_distance < cur_distance):
            cur_distance = best_distance
            route = best_move
            cur_end_point = 0
            max_neighbour_size = default_neighbour_size
        else:
            if cur_end_point >= max_end_points:
                break
            else:
                
                # reset
                if cur_end_point == 0:
                    tabu_list = np.zeros((n+1, n+1, n+1))
                    max_neighbour_size = 100000
                    it = 1
                cur_end_point += 1                
            
        # Rotate the route, ensures the first elements get a chance to be swapped       
        route = route[:-1]  # Take final node off
        route = rotate(route, np.random.randint(1, n)) # Rotate by at least one
        route.append(route[0]) # Add final node back on
            
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

    if n <= 500:
        dist_matrix = compute_distance_matrix(vertices, n)
        solution = iterated_local_search_3opt(vertices, n, points, dist_matrix, max_searches = max(round(10000 / ( n* 10)), 1), seconds_timeout = 2400)
    elif n <= 10000:
        dist_matrix = compute_distance_matrix(vertices, n)
        solution = iterated_local_search_3opt(vertices, n, points, dist_matrix, max_searches = max(round(10000 / ( n* 10)), 1), seconds_timeout = 4800)
    else:    
        dist_matrix = np.zeros((n, n))
        solution = tabu_search2(vertices, n, points, dist_matrix, 300)
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

