import numpy as np
from collections import namedtuple
from queue import Queue


def knight_move(pos1, pos2):
    board = np.zeros((8, 14), dtype=int)
    dir = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    Pos = namedtuple("Pos", ['x', 'y'])

    origin = Pos(int(pos1[1]) - 1, ord(pos1[0]) - 97)
    destination = Pos(int(pos2[1]) - 1, ord(pos2[0]) - 97)

    q = Queue()
    q.put(origin)
    step = 0
    board[origin.x, origin.y] = 1
    rear, cursor = origin, origin
    while not q.empty():
        tmp = q.get()
        if tmp == destination:
            print("To get from {} to {} takes {} knight moves.".format(pos1, pos2, step))
            return

        for i in range(8):
            u = tmp.x + dir[i][0]
            v = tmp.y + dir[i][1]
            if 0 <= u < 8 and 0 <= v < 14 and not board[u, v]:
                next_pos = Pos(u, v)
                q.put(next_pos)
                cursor = next_pos
                board[u, v] = 1

        if tmp == rear:
            rear = cursor
            step += 1

    return -1


if __name__ == "__main__":
    test_data = [("e2", "e4"), ("a1", "b2"), ("b2", "c3"), ("a1", "h8"), ("a1", "h7"), ("h8", "a1"), ("b1", "c3"),
                 ("f6", "f6")]
    for data in test_data:
        knight_move(*data)


