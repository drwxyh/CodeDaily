import numpy as np

INF = 65536


def find_the_last_domino(n, m, edge, edges):
    s = np.zeros(n + 1, dtype=int)
    s[1] = 1
    path = [-1 for i in range(n + 1)]
    path = np.array(path)
    time = [INF for i in range(n + 1)]
    for i in range(1, n + 1):
        time[i] = edge[1, i]
        if i != 1 and time[i] != INF:
            path[i] = 1
        else:
            path[i] = -1

    for i in range(n - 1):
        min_edge = INF
        u = 1
        for j in range(1, n + 1):
            if not s[j] and time[j] < min_edge:
                min_edge = time[j]
                u = j
        s[u] = 1
        for j in range(1, n + 1):
            if not s[j] and edge[u, j] != INF and time[u] + edge[u, j] < time[j]:
                time[j] = time[u] + edge[u, j]
                path[j] = u

    max_vertex_time = 0
    max_vertex = 0
    for i in range(1, n + 1):
        if time[i] > max_vertex_time:
            max_vertex_time = time[i]
            max_vertex = i

    max_edge_time = 0
    edge_pair = [0, 0]
    for e in edges:
        if (time[e[0]] + time[e[1]] + e[2]) / 2 > max_edge_time:
            max_edge_time = (time[e[0]] + time[e[1]] + e[2]) / 2
            edge_pair[0], edge_pair[1] = e[0], e[1]

    if max_edge_time <= max_vertex_time:
        res = "The last domino falls after {:.1f} seconds, at key domino {}."
        print(res.format(max_vertex_time, max_vertex))
    else:
        res = "The last domino falls after {:.1f} seconds, between key dominoes {} and {}."
        print(res.format(max_edge_time, edge_pair[0], edge_pair[1]))


if __name__ == "__main__":
    with open("180912") as fp:
        n, m = [int(x) for x in fp.readline().split()]
        # 下标从 1 开始
        edge = [[INF] * (n + 1) for i in range(n + 1)]
        edge = np.array(edge)
        edges = list()
        for i in range(1, n + 1):
            edge[i, i] = 0
        for line in fp.readlines():
            u, v, w = [int(x) for x in line.split()]
            edge[u, v] = w
            edges.append((u, v, w))

        find_the_last_domino(n, m, edge, edges)


