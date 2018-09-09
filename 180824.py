import numpy as np


def flooring(x, y, used):
    if y == m:
        return flooring(x + 1, 0, used)
    if x == n:
        return 1
    if used[x][y] or color[x][y]:
        return flooring(x, y + 1, used)

    res = 0
    used[x][y] = 1
    if y + 1 < m and not color[x][y + 1] and not used[x][y + 1]:
        used[x][y + 1] = 1
        res += flooring(x, y + 1, used)
        used[x][y + 1] = 0

    if x + 1 < n and not color[x + 1][y] and not used[x + 1][y]:
        used[x + 1][y] = 1
        res += flooring(x, y + 1, used)
        used[x + 1][y] = 0
    used[x][y] = 0

    return res


if __name__ == '__main__':

    with open('180824') as fp:
        n, m = [int(x) for x in fp.readline().split()]
        graph = list()
        for line in fp.readlines():
            graph.append([x for x in line.split()])

    used = np.zeros((n, m), dtype=int)
    color = np.zeros((n, m), dtype=int)
    for i in range(0, n):
        for j in range(0, m):
            if graph[i][j] == 'x':
                color[i][j] = 1
    res = flooring(0, 0, used)
    print(res)
