from collections import namedtuple
from queue import Queue
import numpy as np


def dead_drop():
    q = Queue()
    start = Node(start_pos[0], start_pos[1], 0, hit_point)
    q.put(start)
    visit[start.x][start.y] = 1

    while not q.empty():
        tmp = q.get()

        if (tmp.x, tmp.y) == to_pos:
            print("The minimum number of time that the spy need to fulfill the task is {}.".format(tmp.t))
            return

        for i in range(4):
            u = tmp.x + dir[i][0]
            v = tmp.y + dir[i][1]
            if 0 <= u < row and 0 <= v < col and not visit[u][v] and graph[u][v] != 'x':
                if (graph[u][v] == '.' or graph[u][v] == 'T') and tmp.h - 1 > 0:
                    new_node = Node(u, v, tmp.t + 1, tmp.h - 1)
                elif graph[u][v] == 'w' and tmp.h - 2 > 0:
                    new_node = Node(u, v, tmp.t + 2, tmp.h - 2)
                elif graph[u][v] == 'm' and tmp.h - 1 > 0:
                    new_node = Node(u, v, tmp.t + 3, tmp.h - 1)
                else:
                    continue
                q.put(new_node)
                visit[u][v] = 1

    print("Unfortunately，the poor spy can't fulfill his task.")


if __name__ == "__main__":
    Node = namedtuple("Node", ['x', 'y', 't', 'h'])
    dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    with open("180909") as fp:
        row, col = [int(x) for x in fp.readline().split()]
        graph = list()
        visit = np.zeros((row, col), dtype=int)
        for i in range(row):
            line_data = fp.readline().strip()
            tmp = list()
            for data in line_data:
                tmp.append(data)
            graph.append(tmp)
        hit_point = int(fp.readline())
        for i in range(row):
            for j in range(col):
                if graph[i][j] == 'S':
                    start_pos = (i, j)
                if graph[i][j] == 'T':
                    to_pos = (i, j)

        dead_drop()
