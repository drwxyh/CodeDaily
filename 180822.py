import numpy as np


def find_k_length_path(vertex_num, path_length, adjacency_matrix):
    """
    :param vertex_num: number of vertices in this digraph
    :param path_length: the given path length
    :param adjacency_matrix: adjacency matrix of the digraph
    :return: the number of paths with the given length in this graph
    """
    n, m = vertex_num, path_length
    if m < 1:
        print("Error, The path length is not valid!")
    else:
        arrival_matrix = adjacency_matrix
        for x in range(0, path_length):
            arrival_matrix = np.dot(adjacency_matrix, adjacency_matrix)
        res = 0
        for i in range(0, n):
            for j in range(0, n):
                res += arrival_matrix[i][j]
        print("There are " + str(res) + " paths in this digraph with length " + str(m) + ".")


if __name__ == '__main__':
    with open('180822') as fp:
        vertex_num, path_length = [int(x) for x in fp.readline().split()]
        temp_matrix = list()
        for line in fp.readlines():
            temp_matrix.append([int(x) for x in line.split()])

    adjacency_matrix = np.array(temp_matrix, dtype=int)
    find_k_length_path(vertex_num, path_length, adjacency_matrix)
