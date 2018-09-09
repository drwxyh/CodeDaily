import numpy as np


def knapsack(capacity, item_num, item_cost, item_size):
    """
    :param capacity: the capacity of the knapsack
    :param item_num: the number of the items to be chosen
    :param item_cost: the value of each item
    :param item_size: the size of each item
    :return: the maximum value of the items that the knapsack can load
    """
    A = np.zeros((item_num + 1, capacity + 1), dtype=int)
    for i in range(0, item_num + 1):
        A[i][0] = 0
    for j in range(0, capacity + 1):
        A[0][j] = 0
    for i in range(1, item_num + 1):
        for j in range(1, capacity + 1):
            if j < item_size[i - 1]:
                A[i][j] = A[i - 1][j]
            else:
                A[i][j] = max(A[i - 1][j], A[i - 1][j - item_size[i - 1]] + item_cost[i - 1])

    print("The maximum value of the items that this knapsack can load is : " + str(A[item_num][capacity]) + '$ .')


if __name__ == "__main__":
    with open("180827") as fp:
        capacity = int(fp.readline())
        item_num = int(fp.readline())
        item_cost = [int(x) for x in fp.readline().split()]
        item_size = [int(x) for x in fp.readline().split()]

    knapsack(capacity, item_num, item_cost, item_size)


