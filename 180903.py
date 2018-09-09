from collections import namedtuple
from queue import Queue
import numpy as np


def cover_self(x, y, u, v, state):
    for i in range(1, snake_len):
        k = state & 3
        state >>= 2
        x += dirs[k][0]
        y += dirs[k][1]
        if u == x and v == y:
            return True
    return False


def bfs(x, y, state):
    q = Queue()
    t = Node(x, y, state, 0)
    q.put(t)
    visit[x][y][state] = 1

    while not q.empty():
        t = q.get()

        if t.x == 0 and t.y == 0:
            return t.step

        step = t.step + 1
        for i in range(0, 4):
            u = t.x + dirs[i][0]
            v = t.y + dirs[i][1]
            if u in range(row) and v in range(col) and graph[u][v] != 1 and not cover_self(t.x, t.y, u, v, t.state):
                state = t.state
                state <<= 2
                state |= redirs[i]
                state &= base
                if visit[u][v][state] == 0:
                    cur = Node(u, v, state, step)
                    q.put(cur)
                    visit[u][v][state] = 1

    return -1


if __name__ == "__main__":
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    redirs = [2, 3, 0, 1]
    Node = namedtuple("Node", ['x', 'y', 'state', 'step'])
    with open("180903") as fp:
        row, col, snake_len = [int(x) for x in fp.readline().split()]

        snake_data = list()
        for i in range(snake_len):
            snake_data.append([int(x) - 1 for x in fp.readline().split()])

        stone_num = int(fp.readline())
        stone_data = list()
        for i in range(stone_num):
            stone_data.append([int(x) - 1 for x in fp.readline().split()])

    graph = np.zeros((row, col), dtype=int)
    for i, j in stone_data:
        graph[i][j] = 1

    base = (1 << 2 * (snake_len - 1)) - 1
    visit = np.zeros((row, col, base + 1), dtype=int)

    ans = 1
    for i in range(snake_len - 1, 0, -1):
        for j in range(4):
            if snake_data[i][0] == dirs[j][0] + snake_data[i - 1][0] and snake_data[i][1] == dirs[j][1] + snake_data[i - 1][1]:
                ans <<= 2
                ans |= j
                ans &= base
                break

    a, b = snake_data[0]
    res = bfs(a, b, ans)
    if res == -1:
        print("This poor snake can't not get out of the hole.")
    else:
        print("The minimum number of the steps that the snake should take is " + str(res) + '.')