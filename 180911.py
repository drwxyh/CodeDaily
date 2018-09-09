import numpy as np

INF = 65536


def dijkstra(origin):
    s = np.zeros(node_num, dtype=int)
    s[origin] = 1
    path = np.zeros(node_num, dtype=int)
    dist = np.zeros(node_num, dtype=int)
    for i in range(node_num):
        dist[i] = edge[origin, i]
        if i != origin and dist[i] != INF:
            path[i] = origin
        else:
            path[i] = -1

    for i in range(node_num - 1):
        min_edge = INF
        u = origin
        for j in range(node_num):
            if not s[j] and dist[j] < min_edge:
                u = j
                min_edge = dist[j]

        s[u] = 1
        for k in range(node_num):
            if not s[k] and edge[u][k] != INF and dist[u] + edge[u, k] < dist[k]:
                path[k] = u
                dist[k] = dist[u] + edge[u, k]

    for i in range(node_num):
        if i != origin:
            shortest_path = str(i)
            t = i
            while path[t] != -1:
                shortest_path += '-' + str(path[t])
                t = path[t]

            shortest_path = shortest_path[::-1]
            res = '{} -> {}, the shortest distance is : {:<2} and the shortest path is : {}'
            print(res.format(origin, i, dist[i], shortest_path))


if __name__ == "__main__":
    with open("180911") as fp:
        node_num = int(fp.readline().strip())
        edge = [[INF] * node_num for i in range(node_num)]
        edge = np.array(edge)
        for line in fp.readlines():
            u, v, w = [int(x) for x in line.split()]
            edge[u, v] = w
        for i in range(node_num):
            edge[i, i] = 0

        dijkstra(1)
