import numpy as np

INF = 65532

if __name__ == "__main__":
    with open("180829") as fp:
        node_num = int(fp.readline().strip())
        graph = np.zeros((node_num, node_num), dtype=int)
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    graph[i, j] = 0
                else:
                    graph[i, j] = INF
        for line in fp.readlines():
            u, v, w = [int(x) for x in line.split()]
            graph[u, v] = w

    dp = np.zeros((node_num, 1 << (node_num - 1)), dtype=int)
    for i in range(node_num):
        for j in range(0, 1 << (node_num - 1)):
            dp[i, j] = INF

    for i in range(0, node_num):
        dp[i, 0] = graph[i, 0]

    for i in range(1, 1 << (node_num - 1)):
        for j in range(0, node_num):
            for k in range(1, node_num):
                if 1 << (k - 1) & i:
                    dp[j, i] = min(dp[j, i], graph[j, k] + dp[k, i - (1 << (k - 1))])

    print("The shortest TSP distance is: " + str(dp[0][(1 << node_num - 1) - 1]) + ".")



