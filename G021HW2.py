from os.path import isfile
from random import randint, random
from collections import defaultdict
from argparse import ArgumentParser
from time import time
from pyspark import SparkConf, SparkContext, RDD
from statistics import median


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


def h_(c: int):
    p = 8191
    a = randint(1, p-1)
    b = randint(0, p-1)

    def hash_func(u: int) -> int:
        return ((a*u + b) % p) % c
    
    return hash_func



def MR_ApproxTCwithNodeColors(rdd: RDD, C: int) -> int:
    # set h_c as the proper coloring hash function with C num of colors
    h_c = h_(C)

    def group_by_color(edge):
        """
        Returns the color of edge pairs if being in the same color, otherwise -1
        """
        v1, v2 = edge
        c1, c2 = h_c(v1), h_c(v2)  # evaluate colors of the two vertices
        return [(c1, (v1, v2))] if c1==c2 else []

    t_final = (
        rdd .flatMap(group_by_color)
            .groupByKey()  # E(i)
            .map(lambda group: (group[0], count_triangles(group[1])))  # t(i)
            .values().sum() * C**2  # t_final
    )
    return t_final

def MR_ExactTC(rdd: RDD, C: int) -> int:
    h_c = h_(C)
    
    def get_triplet(edge):
        u, v = edge
        hc_u, hc_v = h_c(u), h_c(v)  # evaluate colors of the two vertices
        return [((hc_u, hc_v, i), (u, v)) for i in range(C)]
    
    # Target: 1612010
    rdd = rdd.flatMap(get_triplet).groupByKey().map(lambda group: (group[0], count_triangles(group[1])))
    print('keys',rdd.keys().collect())
    print(rdd.collect())
    print(rdd.values().collect())
    return rdd.values().sum()


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
    assert isfile(args.path), "Invalid data file path (argument FILE_PATH)"

    # Spark configuration
    conf = SparkConf().setAppName("BDC:G021HW2")
    sc = SparkContext(conf=conf)

    # Reading dataset to RDD
    rdd = sc.textFile(args.path, minPartitions=args.C, use_unicode=False)
    rdd = rdd.map(lambda s: tuple(map(int, s.split(b',')))) # Convert edges from string to tuple
    rdd = rdd.partitionBy(args.C, lambda _:randint(0, args.C-1)) # Random partitioning instead of repartition()
    rdd = rdd.cache()
    
    print("Dataset =", args.path.replace('\\','/').split('/')[-1])
    print("Number of Edges =", rdd.count())
    print("Number of Colors =", args.C)
    print("Number of Repetitions =", args.R)

    if args.F == 0:
        print("Approximation through node coloring")
        t_final_list = []
        total_times_list = []
        for _ in range(args.R):
            start_time = time()
            t_final = MR_ApproxTCwithNodeColors(rdd, args.C)
            end_time = time()

            total_times_list.append(end_time-start_time)
            t_final_list.append(t_final)
        print(f"- Number of triangles (median over {args.R} runs) = {median(t_final_list)}")
        print(f"- Running time (average over {args.R} runs) = {median(total_times_list)*1000:.0f} ms")
    else:
        print("Approximation ExactTC")
        t_final_list = []
        total_times_list = []
        for _ in range(args.R):
            start_time = time()
            t_final = MR_ExactTC(rdd, args.C)
            end_time = time()

            total_times_list.append(end_time-start_time)
            t_final_list.append(t_final)
        print(f"- Number of triangles (median over {args.R} runs) = {median(t_final_list)}")
        print(f"- Running time (average over {args.R} runs) = {median(total_times_list)*1000:.0f} ms")
    
if __name__ == '__main__':
    main()
