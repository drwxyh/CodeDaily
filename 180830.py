def find_non_irrigated_grid(graph, row, col):
    for i in range(row):
        for j in range(col):
            if graph[i][j] != ".":
                return i, j
    return None


def is_connected(a, b, d):
    flag = 0
    if d == 0:
        if grid[a][1] == '1' and grid[b][3] == '1':
            flag = 1
    elif d == 1:
        if grid[a][3] == '1' and grid[b][1] == '1':
            flag = 1
    elif d == 2:
        if grid[a][2] == '1' and grid[b][0] == '1':
            flag = 1
    else:
        if grid[a][0] == '1' and grid[b][2] == '1':
            flag = 1
    return flag


def dfs(x, y):
    temp = graph[x][y]
    graph[x][y] = '.'
    for i in range(4):
        u = x + dir[i][0]
        v = y + dir[i][1]
        if u < 0 or u >= row or v < 0 or v >= col:
            continue
        if graph[u][v] == '.':
            continue
        if is_connected(temp, graph[u][v], i):
            dfs(u, v)


if __name__ == "__main__":
    with open("180830") as fp:
        row, col = [int(x) for x in fp.readline().split()]
        graph = list()
        for data in fp.readlines():
            line = [x for x in data]
            graph.append(line)

    dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    grid = {'A': '1001', 'B': '1100', 'C': '0011', 'D': '0110', 'E': '1010', 'F': '0101', 'G': '1101', 'H': '1011',
            'I': '0111', 'J': '1110', 'K': '1111'}

    res = find_non_irrigated_grid(graph, row, col)
    cnt = 0
    while res:
        cnt += 1
        x, y = res
        dfs(x, y)
        res = find_non_irrigated_grid(graph, row, col)
    if cnt == 1:
        print("Totally one water source is needed.")
    else:
        print("Totally " + str(cnt) + " headwaters are needed.")




