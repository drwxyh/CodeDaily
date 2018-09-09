import numpy as np
from collections import namedtuple
from queue import Queue


def find_the_smallest_multiple():
    q = Queue()
    visit = np.zeros(n, dtype=int)
    Node = namedtuple("Node", ["value", "remainder"])
    s = Node(0, 0)
    q.put(s)

    while not q.empty():
        t = q.get()

        for x in nums:
            k = t.remainder * 10 + x
            if not k:
                continue
            else:
                k %= n
                if k == 0:
                    print(t.value * 10 + x)
                    return
                if visit[k] == 0:
                    visit[k] = 1
                    new_node = Node(t.value * 10 + x, k)
                    q.put(new_node)
    print(0)


if __name__ == "__main__":
    with open("180825") as fp:
        n = int(fp.readline().strip())
        m = int(fp.readline().strip())
        nums = sorted([int(x) for x in fp.readlines()])

        find_the_smallest_multiple()
