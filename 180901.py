from collections import namedtuple


def dfs(cur):
    if cur == side_length ** 2:
        return True

    for i in range(len(square_kind)):
        square = square_kind[i]
        if square_dict[square] == 0:
            continue

        if cur > 0 and cur % side_length:
            if square_kind[board[cur - 1]].r != square.l:
                continue

        if cur > side_length:
            if square_kind[board[cur - side_length]].d != square.u:
                continue

        board[cur] = i
        square_dict[square] -= 1
        if dfs(cur + 1):
            return True
        else:
            square_dict[square] += 1

    return False


if __name__ == "__main__":
    Square = namedtuple("Square", ['u', 'r', 'd', 'l'])
    square_list = list()
    with open("180901") as fp:
        side_length = int(fp.readline())
        for data in fp.readlines():
            u, r, d, l = [int(x) for x in data.split(' ')]
            square_list.append(Square(u, r, d, l))

    square_dict = dict()
    for square in square_list:
        if square not in square_dict.items():
            square_dict[square] = 1
        else:
            square_dict[square] += 1

    square_kind = [x for x in square_dict.keys()]
    board = [-1 for i in range(side_length ** 2)]
    if dfs(0):
        print("Possible.")
    else:
        print("Impossible.")


