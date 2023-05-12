from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict
from time import time
import statistics



def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count

def rawData_to_edges(rawData):
    vertexes = rawData.split(",", 1)
    v = int(vertexes[0])
    u = int(vertexes[1])
    return [[u,v]]

def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    #We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity


    colors = list(colors_tuple)
    #Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    #Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:

        u, v = edge
        node_colors[u]= ((rand_a*u+rand_b)%p)%num_colors
        node_colors[v]= ((rand_a*v+rand_b)%p)%num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                        # If triangle has the right colors, count it.
                        if colors==triangle_colors:
                            triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count

def MR_ApproxTCwithNodeColors(edges,c):
    p = 8191
    a = rand.randint(1, p - 1)
    b = rand.randint(0, p - 1)


    def hash_function(edge):
        v,u = edge

        hash_codeV1 = ((a * v + b) % p) % c
        hash_codeV2 = ((a * u + b) % p) % c
        if hash_codeV1 == hash_codeV2:
            return [(hash_codeV1,edge)]
        return []


    triangle_counting = (edges.flatMap(hash_function) # <-- MAP PHASE (0,(2000,2001))
                .groupByKey() # (0,[(2000,2001),(2009,2008),...]
                .mapValues(CountTriangles)  # (0,2200) , (1,2100) , (2,2000),3(
                .values()
                .sum()*c*c
                )
    return triangle_counting

def MR_ExactTC(edges,c):
    p = 8191
    a = rand.randint(1, p - 1)
    b = rand.randint(0, p - 1)

    def keytozero(pair):
        return [(0, pair[1])]

    def generatePairs(edge):
        v,u = edge

        hash_codeV1 = ((a * v + b) % p) % c
        hash_codeV2 = ((a * u + b) % p) % c

        #listOFKeyVal = []
        return [(tuple(sorted((hash_codeV1,hash_codeV2,i))),edge) for i in range(c)]



    triangle_counting = (edges.flatMap(generatePairs)
                .groupByKey()
                .map(lambda keyListOfVals: (keyListOfVals[0],countTriangles2(keyListOfVals[0],keyListOfVals[1],a,b,p,c)))
                .values()
                .sum()
                )
    return triangle_counting




def main():
    assert len(sys.argv) == 4, "Usage: python Triangle_Counting.py <C> <R> <file_name>"
    conf = SparkConf().setAppName("Triangle_Counting")
    sc = SparkContext(conf=conf)

    #input reading

    #1. read number of partitions
    C = sys.argv[1]
    assert C.isdigit(), "k must be an integer"
    C = int(C)

    R = sys.argv[2]
    assert R.isdigit(), "k must be an integer"
    R = int(R)

    #2. read input file and subdivide it into k random partitions
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path).repartition(numPartitions=32).cache()
    edges = rawData.flatMap(rawData_to_edges)
    #docs.repartition(numPartitions=c)

    #setting global variables

    Number_of_triangles = []
    Number_of_triangles_spark = []
    avg_running_time1 = 0
    avg_running_time2 = 0
    cur_runtime = 0
    '''
    for i in range(R):
        start_time = time()
        Number_of_triangles.append(MR_ApproxTCwithNodeColors(edges,C))
        cur_runtime = (time() - start_time)*1000
        avg_running_time1+= cur_runtime
    avg_running_time1 = avg_running_time1/R

    


    print("Dataset = " + data_path)
    print("Number of Edges = " + str(edges.count()))
    print("Number of Colors = " + str(C))
    print("Number of Repetitions = " + str(R))

    print("Approximation through node coloring")
    print("- Number of Triangle (median over " + str(R) + " runs) =", statistics.median(Number_of_triangles))
    print("- Running time (average over " + str(R) + " runs) = ", avg_running_time1)
    '''
    print("Dataset = " + data_path)
    print("Number of Edges = " + str(edges.count()))
    print("Number of Colors = " + str(C))
    print("Number of Repetitions = " + str(R))

    print("Approximation through node coloring")
    print("- Number of Triangle (median over " + str(R) + " runs) =", MR_ExactTC(edges,C))
    #print("- Running time (average over " + str(R) + " runs) = ", avg_running_time1)


    # print("Number of Triangle in the graph =", Number_of_triangles_spark[0].collect().__len__())#.collect())



if __name__ == '__main__':
    main()