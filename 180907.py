from collections import namedtuple
from queue import Queue


def basic_maze():
    q = Queue()
    q.put(origin)
    step = 0
    rear, cursor = origin, origin
    while not q.empty():
        tmp = q.get()

        if tmp == destination:
            print("The minimum number of steps that needs to take is: {}.".format(step))
            print_path()
            return

        for i in range(4):
            u = tmp.x + dir[i][0]
            v = tmp.y + dir[i][1]
            if 0 <= u < 6 and 0 <= v < 6 and not graph[u][v] and not is_cross_the_wall(tmp, Pos(u, v)):
                graph[u][v] = (tmp.x, tmp.y)
                next_node = Pos(u, v)
                q.put(next_node)
                cursor = next_node

        if tmp == rear:
            step += 1
            rear = cursor


def print_path():
    res = ""
    action = ['E', 'S', 'W', 'N']
    x = destination.x
    y = destination.y
    while graph[x][y] != "origin":
        for i in range(4):
            if x == graph[x][y][0] + dir[i][0] and y == graph[x][y][1] + dir[i][1]:
                res += action[i]
        x, y = graph[x][y][0], graph[x][y][1]
    print("The route is: {}.".format(res[::-1]))


def is_cross_the_wall(a, b):
    for i in range(3):
        if walls[i].s.x == walls[i].t.x:
            if walls[i].s.x == 0 or walls[i].s.x == 6:
                continue
            else:
                if a.y == b.y and a.y in range(walls[i].s.y, walls[i].t.y):
                    if (a.x < walls[i].s.x and b.x == walls[i].s.x) or (b.x < walls[i].s.x and a.x == walls[i].s.x):
                        return True

                else:
                    continue

        if walls[i].s.y == walls[i].t.y:
            # 位于边界的墙不影响
            if walls[i].s.y == 0 or walls[i].s.y == 6:
                continue
            else:
                # 纵向
                if a.x == b.x and a.x in range(walls[i].s.x, walls[i].t.x):
                    if (a.y < walls[i].s.y and b.y == walls[i].s.y) or (b.y < walls[i].s.y and a.y == walls[i].s.y):
                        return True
                    else:
                        continue
    return False


if __name__ == "__main__":
    graph = [[0] * 6 for i in range(6)]
    dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    Pos = namedtuple("Pos", ['x', 'y'])
    Wall = namedtuple("Wall", ['s', 't'])
    walls = list()

    with open("180907") as fp:
        y, x = [int(x) - 1 for x in fp.readline().split()]
        origin = Pos(x, y)
        graph[origin.x][origin.y] = "origin"
        y, x = [int(x) - 1 for x in fp.readline().split()]
        destination = Pos(x, y)
        for i in range(3):
            pos1_y, pos1_x, pos2_y, pos2_x = [int(x) for x in fp.readline().split()]
            walls.append(Wall(Pos(pos1_x, pos1_y), Pos(pos2_x, pos2_y)))

    basic_maze()
