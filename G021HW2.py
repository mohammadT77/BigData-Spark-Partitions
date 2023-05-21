# from os.path import isfile
from random import randint
from collections import defaultdict
from argparse import ArgumentParser
from time import time
from statistics import median, mean
from pyspark import SparkConf, SparkContext, RDD


def count_triangles(edges) -> int:
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


def count_triangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
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



def h_(c: int):
    # Returning hash function after configuring a, b variables randomly
    p = 8191
    a = randint(1, p-1)
    b = randint(0, p-1)

    def hash_func(u: int) -> int:
        return ((a*u + b) % p) % c

    # set needed variables a,b,p to the function for future uses
    hash_func.a = a
    hash_func.b = b
    hash_func.p = p

    return hash_func



def MR_ApproxTCwithNodeColors(edges: RDD, C: int) -> int:
    # set h_c as the proper coloring hash function with C num of colors
    h_c = h_(C)

    def group_by_color(edge):
        # Returns the color of edge pairs if being in the same color, otherwise -1
        v1, v2 = edge
        c1, c2 = h_c(v1), h_c(v2)  # evaluate colors of the two vertices
        return [(c1, (v1, v2))] if c1==c2 else []

    t_final = (
        edges.flatMap(group_by_color)
            .groupByKey()  # E(i)
            .map(lambda group: (group[0], count_triangles(group[1])))  # t(i)
            .values().sum() * C**2  # t_final
    )
    return t_final


def MR_ExactTC(edges: RDD, C: int) -> int:
    h_c = h_(C)  # instantiate a hash function

    p, a, b = h_c.p, h_c.a, h_c.b

    def generate_pairs(edge):
        u, v = edge
        hc_u, hc_v = h_c(u), h_c(v)  # evaluate the colors
        return [(tuple(sorted((hc_u, hc_v, i))), (u, v)) for i in range(C)]

    t_final = (
        edges.flatMap(generate_pairs)
            .groupByKey()
            .map(lambda item: (item[0], count_triangles2(item[0],item[1], a, b, p, C)))
            .values()
            .sum()
    )
    return t_final


def main():
    # Configure argument parser
    parser = ArgumentParser(description="BDC - Group 021 - Assignment 2")

    parser.add_argument('C', type=int, help='Number of colors')
    parser.add_argument('R', type=int, help='Number of Repetitions')
    parser.add_argument('F', type=int, help='Number of Repetitions')
    parser.add_argument('path', metavar="FILE_PATH", type=str, help='Dataset file path')

    args = parser.parse_args()

    # Validate arguments
    assert args.C >= 1, "Invalid argument C"
    assert args.R >= 1, "Invalid argument R"
    assert args.F in (0,1), "Invalid argument F"
    # assert isfile(args.path), "Invalid data file path (argument FILE_PATH)"

    # Spark configuration
    conf = SparkConf().setAppName("BDC:G021HW2").set('spark.locality.wait', '0s')
    sc = SparkContext(conf=conf)
    # sc.setLogLevel('WARN')

    # Reading dataset to RDD
    rawData = sc.textFile(args.path, minPartitions=args.C, use_unicode=False)
    edges = rawData.map(lambda s: tuple(map(int, s.split(b',')))) # Convert edges from string to tuple
    edges = edges.repartition(32)
    edges = edges.cache()

    print("Dataset =", args.path)
    print("Number of Edges =", edges.count())
    print("Number of Colors =", args.C)
    print("Number of Repetitions =", args.R)

    if args.F == 0:
        print("Approximation algorithm with node coloring")
        t_final_list = []
        total_times_list = []
        for _ in range(args.R):
            start_time = time()
            t_final = MR_ApproxTCwithNodeColors(edges, args.C)
            end_time = time()

            total_times_list.append(end_time-start_time)
            t_final_list.append(t_final)
        print("- Number of triangles (median over {} runs) = {}".format(args.R, median(t_final_list)) )
        print("- Running time (average over {} runs) = {} ms".format(args.R, int(mean(total_times_list)*1000)) )
    else:
        print("Exact algorithm with node coloring")
        t_final_list = []
        total_times_list = []
        for _ in range(args.R):
            start_time = time()
            t_final = MR_ExactTC(edges, args.C)
            end_time = time()

            total_times_list.append(end_time-start_time)
            t_final_list.append(t_final)
        print("- Number of triangles (median over {} runs) = {}".format(args.R, t_final_list[-1]))
        print("- Running time (average over {} runs) = {} ms".format(args.R, int(mean(total_times_list)*1000)))


if __name__ == '__main__':
    # Handling thrown exceptions from main()
    try:
        main()
    except Exception as e:
        print(e.__class__.__name__, "occurred:", e)

