def find_oil():
    for i in range(row):
        for j in range(col):
            if field[i][j] == '@':
                return i, j


def dfs(x, y):
    field[x][y] = '.'
    dir = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i in range(8):
        u = x + dir[i][0]
        v = y + dir[i][1]
        if 0 <= u < row and 0 <= v < col and field[u][v] == '@':
            dfs(u, v)


if __name__ == "__main__":
    with open("180828") as fp:
        row, col = [int(x) for x in fp.readline().split()]
        field = list()
        for line in fp.readlines():
            tmp = list()
            for c in line.strip():
                tmp.append(c)
            field.append(tmp)

    cnt = 0
    res = find_oil()
    while res:
        cnt += 1
        dfs(res[0], res[1])
        res = find_oil()
    if cnt == 1:
        print("There is only one oil field.")
    else:
        print("There are {} oil fields.".format(cnt))
